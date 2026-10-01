# Verification log

What was actually run, and what it produced. This file exists because a project
that claims numbers should be able to show the command that produced them.

**Last run:** 2026-10-02, on Windows 11, Python 3.12.10, Node v24.14.0,
Postgres 16 and Redis 7 in Docker.

---

## Table of contents

- [Automated suites](#automated-suites)
- [Live API](#live-api)
- [Five bugs this found](#five-bugs-this-found)
- [What is *not* verified](#what-is-not-verified)
- [Reproducing this](#reproducing-this)

---

## Automated suites

```bash
cd backend  && ../.venv/Scripts/python.exe -m pytest      # 422 passed in 17.42s
cd frontend && npx vitest run                             # 152 passed in 6 files
cd frontend && npm run typecheck                          # clean
cd frontend && npm run lint                               # clean
cd backend  && ../.venv/Scripts/python.exe -m ruff check app tests   # All checks passed!
```

**422 backend tests** across 14 files. **152 frontend tests** across 6 files.
Frontend and backend both include the tenancy suite; the backend count grew from
416 to 422 when `test_route_contracts.py` was added.

Integration tests run against a real Postgres and Redis, driving the actual ASGI
app through `httpx.ASGITransport` — nothing about the database is mocked.

---

## Live API

Backend on `:8000`, exercised with `curl`.

| Check | Result |
|---|---|
| `GET /api/health` | 200 in ~13 ms, `status: degraded` |
| `GET /api/applications` | 200 |
| `GET /api/dashboard/overview?window=90d` | 200, ~500–600 ms cold |
| `GET /api/traces` | 200 |
| `GET /api/evaluations/runs` | 200 |
| `GET /api/recommendations` | 200 |
| `GET /api/anomalies` | 200 |
| `POST /api/traces` | 201, trace id echoed back |
| `POST /api/recommendations/generate` | 200, 6 rules evaluated |
| `POST /api/evaluations/answer` | 200, real metrics, `judge_*` all `null` |
| `POST /api/evaluations/retrieval` without a key | 401, as designed |

The health check reports `degraded` on exactly one component — `vector_store` —
because the knowledge base has not been indexed. Postgres and Redis are both
reachable, and all 18 tables are present. See *What is not verified* below.

One `404` appeared on the first `POST /api/recommendations/generate` after a
database reset. That was correct behaviour, not a fault: the request named an
application that did not exist yet, and it was created by the `POST /api/traces`
that followed 40 ms later. The uvicorn log shows `application.created` between
the two. It is recorded here because a single 404 that then returns 200 five
times is worth explaining rather than forgetting.

---

## Five bugs this found

All five were found by driving the running API, not by reading code. All five
survived a fully green test suite, because no test exercised those paths. Each
now has a regression test in `backend/tests/test_route_contracts.py`.

| # | Bug | Symptom | Root cause |
|---|---|---|---|
| 1 | `compare_runs` the handler shadowed `compare_runs` the import | `POST /evaluations/compare` recursed and 500'd | The name bound in the module namespace, so the call inside the body resolved to the handler |
| 2 | Fabricated `WindowFilter` | `TypeError` on *every* call | `start` and `end` are required; a windowless lookup was inventing two datetimes to carry an org id |
| 3 | Four routes typed `principal: Principal` | 422 for every caller | `Principal` is a plain frozen dataclass; written bare, FastAPI had no `Depends` to resolve and read it as a required **body field** |
| 4 | `_close_run` handed an ORM row where it wanted `.id` | 500 at the end of an otherwise successful batch | SQLAlchemy tried to inline a whole model as a SQL literal |
| 5 | `_close_run` aggregated over unflushed rows | Run stored `num_items: 0`, every metric `0.0` | `SessionLocal` sets `autoflush=False`, so the `SELECT` queried *around* the rows the same transaction had added |

Bug 5 is the one worth remembering. Nothing crashed. The endpoint returned 200,
the per-item `answer_evaluations` rows committed correctly, and the response
body carried the correct numbers. Only the *stored* aggregate was wrong — and
only a caller comparing the run list against the run's own detail page would
notice. It was found by making exactly that comparison.

The first three are shape bugs: they compile, they import, and `ruff` does not
flag them. That is why the regression tests assert over parsed `ast` rather than
calling the endpoints — a behavioural test for bug 3 would need to construct a
valid `Principal` body, which would pass rather than fail.

The unflushed run `59808dcb` was deliberately **left** with its zeros rather
than back-filled. Rewriting it would make the database disagree with what
actually happened.

---

## What is *not* verified

Stated rather than omitted, because a verification log that only lists wins is
marketing.

- **`mypy` reports 58 errors.** `make lint` runs it and it fails. It has never
  been brought to zero. `ruff check` is clean; `mypy` is not, and the two are
  not the same gate.
- **`vector_store` is degraded.** `/api/health` reports it, correctly. The
  knowledge base has not been indexed, so the RAG demo path has not been
  exercised through the API in this log.
- **`langgraph` is not installed.** `evaluation/agent_demo/app.py` has an
  honest built-in fallback runner, so `make demo-agent` runs — but it runs the
  fallback, not LangGraph. The dependency should be declared or the fallback
  documented as the supported path.
- **The LLM judge has never returned a score.** Every `judge_*` field in this
  log is `null`, which proves the quarantine but not the judge. The judge path
  is exercised structurally, not behaviourally.
- **No load or concurrency testing.** Every measurement here is a single
  request. The ~500 ms cold overview has not been characterised under
  concurrency.
- **One platform.** Windows, Git Bash, Docker Desktop. The Makefile assumes a
  POSIX shell; the PowerShell path in the README is documented but has not been
  run end to end here.
- **`scripts/dev.ps1`** is referenced in the Makefile header as the
  PowerShell equivalent. It is not what these runs used.

---

## Reproducing this

```bash
python -m venv .venv      # must come first: `make install` runs .venv/Scripts/pip.exe
make install && make infra
make migrate && make seed
make demo-data        # so the dashboard is not empty
make backend          # terminal 1
make frontend         # terminal 2

# suites
cd backend  && ../.venv/Scripts/python.exe -m pytest
cd frontend && npx vitest run
```

`docs/api-contract.json` is regenerated from the running app's own OpenAPI
schema with `make api-contract`, so it describes what the server does rather
than what someone intended.
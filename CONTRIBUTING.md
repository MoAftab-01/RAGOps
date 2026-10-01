# Contributing to RAGOps

Thanks for looking at this. The project is small and the rules are short, so
this page should be enough to get a change merged.

---

## Table of contents

- [Getting set up](#getting-set-up)
- [Before you open a pull request](#before-you-open-a-pull-request)
- [The three rules](#the-three-rules)
- [Testing conventions](#testing-conventions)
- [Commit and PR conventions](#commit-and-pr-conventions)
- [What needs a test](#what-needs-a-test)
- [Secrets](#secrets)
- [Reporting a bug](#reporting-a-bug)

---

## Getting set up

```bash
git clone <your-fork-url> ragops && cd ragops

python -m venv .venv                                   # the venv must exist first
make install                                          # backend + ML dependencies
make infra                                            # Postgres + Redis (+ Qdrant)
cp .env.example .env
make migrate
make seed
make backend     # terminal 1
make frontend    # terminal 2
```

`make install` installs into `.venv` but does **not** create it — the `PY` and
`PIP` variables point at `.venv/Scripts/*.exe`, so the venv has to exist before
that target is useful.

If you are not on a POSIX shell, see the Windows note in the
[README](README.md#windows-note) for the raw commands.

**You do not need a `.env` to run the tests.** Every default in
`backend/app/config.py` points at the Docker Compose stack, so a clean checkout
plus `make infra` is enough. A `.env` is only needed when running outside Docker
or changing a default.

---

## Before you open a pull request

```bash
make verify
```

That migrates the database and runs both test suites (`verify` = `migrate` +
`test`). It does **not** run lint, so run that too:

```bash
cd backend && ../.venv/Scripts/python.exe -m ruff check app tests
cd frontend && npm run typecheck && npm run lint
```

If something fails, the PR does not go in. If something legitimately cannot
run on your machine, say so in the description rather than skipping it silently.

> **Known gap:** `make lint` also runs `mypy app`, which currently reports 58
> errors. This is not enforced in CI because it is not clean. Do not introduce
> new ones, and do not treat a green `mypy` as a precondition until the existing
> backlog is cleared.

---

## The three rules

These are the invariants the codebase is organised around. A change that breaks
one is a change that needs a different design, not a workaround.

### 1. An unmeasured thing is never zero

Zero is a score. `None` means "not run." A disabled LLM judge must report
`null`, never `0.0` — a judge that did not run and a judge that scored your
batch at zero are completely different findings, and conflating them makes the
second invisible.

The same applies to unevaluated experiment arms, unlabelled dataset rows, and
any metric a run did not record. If a value does not exist, store `None` or
count it explicitly.

### 2. No metric is computed by an LLM

Precision, Recall, MRR, nDCG, token counts, cost, and latency are arithmetic
over recorded values. The one exception is the optional LLM judge, whose output
is quarantined in the `judge_*` fields and can never fall back into the
deterministic columns.

This also rules out using an LLM to *approximate* something an LLM should not
be doing at all — token counting is a string operation, not a reasoning task.

### 3. No business logic in routes, and all SQL through the ORM

A route resolves the application, validates the request, delegates, and
returns. Anything reusable goes in a service. Every `select()` goes in a
repository, and tenant scoping goes through `apply_tenant_filter` (or
`tenant_predicates` for a lookup with no time window) so "unscoped" has exactly
one meaning.

There is no string-built SQL anywhere. This is a SQL-injection boundary, not a
style preference.

---

## Testing conventions

```bash
make test-backend    # pytest, real Postgres + Redis
make test-frontend   # vitest

cd backend  && ../.venv/Scripts/python.exe -m ruff check app tests   # what make lint runs, minus mypy
cd frontend && npm run typecheck && npm run lint
```

**Backend tests run against the real thing.** Integration tests drive the
actual ASGI app through `httpx.ASGITransport` with a managed session, not a
mocked database. A test that mocks the ORM proves nothing about the SQL.

**Frontend tests use the real components** with the API client spied via
`vi.spyOn` rather than replaced by a mock module, so a change to the client's
shape breaks the tests instead of hiding from them.

### Structural guards exist on purpose

Two test files assert over parsed source rather than calling endpoints:

| Test | What it prevents |
|---|---|
| `tests/test_scoping.py` | A query added without a tenant filter, which fails silently as an empty dashboard rather than an error |
| `tests/test_route_contracts.py` | A route typing `principal: Principal` instead of `TenantPrincipal`, which turns an injected parameter into a required body field and 422s every caller |

They are unusual because the defects they guard are *shape* bugs: they compile,
they import, and a green suite does not catch them. If you add a route, they
cover it automatically.

### The bar for a new test

A test should fail before the fix and pass after it. Two failure modes to avoid:

- **Asserting the shape of the output rather than its meaning.** `assert metrics["num_items"] > 0` passes on any batch whose true mean is 0.0. Assert against the specific numbers.
- **Reaching the DB to check what the code wrote** when the API can tell you. Read the number back through the endpoint that renders it — that is the one that has to be right.

---

## Commit and PR conventions

- **Branch first.** Do not commit to `main`.
- **One concern per PR.** A refactor and a behaviour change in the same diff is unreviewable.
- **Write the commit message for someone who will read it in six months.** What broke, why, and what you decided.
- **In the description, say what you verified** — the command you ran and its result. "Should work" is not a verification.

---

## What needs a test

Everything, but especially:

- **Tenancy.** Any new read path or query. A missing tenant filter does not fail; it returns someone else's rows.
- **Metric definitions.** Precision's denominator is `k`, not the number retrieved. If you change a formula, say so in the docstring and add the case that distinguishes it from the obvious wrong one.
- **Judge quarantine.** Anything that touches `judge_*`.
- **Aggregation over unflushed state.** See the note in `answer_eval.py::_close_run`; `autoflush=False` makes this subtle enough to document.

---

## Secrets

**Never commit `.env`.** It is git-ignored, and it should stay that way.

- All configuration is environment variables.
- `.env.example` documents every option, commented out, with no real values.
- API keys are stored as HMAC-SHA256 digests and returned in plaintext exactly once, at creation. If you need to test with a key, **assemble it in the test** rather than pasting a real one — a test file is somewhere a real credential ends up committed.
- No endpoint returns a key after creation. `GET` returns `key_prefix` only.
- Log the key's prefix if you must, never the key.

---

## Reporting a bug

A good report has:

1. What you did, exactly — the command or the request body.
2. What you expected.
3. What happened, including the response body and the server log line.
4. Whether it reproduces on a clean checkout.

If it is a security issue, do not open a public issue. Say so privately to the
maintainers instead.
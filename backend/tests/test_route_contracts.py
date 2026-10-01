"""Regression tests for five defects that no test covered.

Every test in this file exists because something was broken for a long time and
nothing noticed. They all share a shape: the defect was invisible to a unit test
because it lived at the seam between the route signature and FastAPI's
dependency injection, or between an unflushed session and a read-back.

The five, in the order they were found by driving the API with curl:

1. ``compare_runs`` the handler shadowed ``compare_runs`` the import, so
   ``POST /evaluations/compare`` recursed into itself and 500'd.
2. ``WindowFilter(start=..., end=...)`` fabricated to carry an organization id
   through ``apply_tenant_filter`` raised ``TypeError`` on *every* call, because
   ``start`` and ``end`` are required.
3. Four routes declared ``principal: Principal`` -- the raw Pydantic class --
   instead of the ``Annotated[..., Depends(current_principal)]`` alias. FastAPI
   read it as a required *body field*, so every one of them 422'd for any caller
   who did not send a ``Principal`` blob.
4. ``_close_run`` was handed the ORM row where it wanted ``.id``, and the route
   500'd at the very end of an otherwise successful batch.
5. ``_close_run`` aggregated over a SELECT that could not see the rows the same
   transaction had added, because ``SessionLocal`` sets ``autoflush=False``. The
   individual rows committed; the run's headline numbers read ``num_items: 0``
   with every metric ``0.0``.

Tests 1-4 are about the *shape* of a route, so they are asserted statically
against the parsed source rather than by calling the endpoint -- an HTTP call
would need a fixture that fails the test for the wrong reason. Test 5 is a
behavioural bug and is driven over HTTP, because a real aggregate is the only
honest way to prove the flush happened.

The rule throughout: a guard that fails must name the file and line, so the
failure is actionable without a debugger.
"""

from __future__ import annotations

import ast
import pathlib
import uuid
from typing import Any

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("shared_key_enabled"),
]

#: ``app/`` -- the importable package, not the ``backend/`` directory, so the
#: paths in a failure read the way a developer would type them.
APP_ROOT = pathlib.Path(__file__).resolve().parent.parent / "app"

#: The shared key ``shared_key_enabled`` installs, and therefore the one that
#: resolves to the platform principal. Used by the behavioural tests below,
#: which are not tenancy tests: answer evaluation is not organization-scoped, and
#: these assertions are about flushes and judge quarantine rather than about
#: isolation. Nothing here prints it, and it is a fixture constant, not a
#: credential -- see ``test_tenancy.py`` for why that distinction is structural.
PLATFORM_HEADERS = {"X-API-Key": "test-shared-key"}

#: Every route module. The structural guards below walk exactly this set, so a
#: new router is covered the moment it is added rather than the next time
#: someone remembers to list it.
ROUTER_DIR = APP_ROOT / "api" / "v1"


def _route_modules() -> list[pathlib.Path]:
    return sorted(p for p in ROUTER_DIR.glob("*.py") if not p.name.startswith("_"))


def _is_route_handler(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when a decorator registers this function on a router.

    Specifically ``@router.get(...)`` / ``@router.post(...)`` -- an attribute
    call on something named ``router``. This distinction is the whole point of
    the guard below and getting it wrong produces the opposite of a useful
    failure: ``organizations.py`` has a plain helper called ``_require_platform``
    that legitimately takes ``principal: Principal``, because it is *called by* a
    route and never called by FastAPI. Flagging it would mean silencing the guard
    to accommodate a false positive, which loses the real check.
    """
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "router"
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# 1 + 3. Route signatures
# ---------------------------------------------------------------------------


def test_no_route_declares_a_bare_principal() -> None:
    """No route may type a principal as ``Principal`` (defect 3).

    ``Principal`` is a Pydantic model. Written into a route signature as
    ``principal: Principal`` it is not an injection at all -- FastAPI reads it as
    a required request-body field, and the endpoint then rejects every caller
    with ``422 {"loc": ["body", "principal"], "msg": "Field required"}``.

    The injection is ``TenantPrincipal``, defined once in ``security.py`` as
    ``Annotated[Principal, Depends(current_principal)]``. That alias is the only
    spelling that works, and the only one a reader can check by eye.

    Scoped to functions a router actually registers. A *plain helper* is free to
    take ``principal: Principal`` -- it is called by the route, not by FastAPI --
    and ``organizations.py::_require_platform`` does exactly that.

    Detected over the parsed AST rather than by grep so that a *docstring* or a
    comment mentioning the bare form -- both of which exist in this codebase to
    explain the trap -- cannot trip the guard, and so the failure can name the
    exact line.
    """
    offenders: list[tuple[str, int, str]] = []

    for path in _route_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not _is_route_handler(node):
                continue
            for arg in (*node.args.args, *node.args.kwonlyargs):
                # The annotation text, or "" for a bare `principal` with no type.
                annotation = ast.unparse(arg.annotation) if arg.annotation else ""
                if arg.arg == "principal" and annotation == "Principal":
                    offenders.append(
                        (
                            path.relative_to(APP_ROOT.parent).as_posix(),
                            arg.lineno,
                            node.name,
                        )
                    )

    assert not offenders, (
        "A route types its principal as the raw `Principal` Pydantic model, which "
        "FastAPI treats as a required request-body field rather than an "
        "injection. Use `TenantPrincipal` from `app.api.deps`:\n"
        + "\n".join(f"  {file}:{line} in {func}()" for file, line, func in offenders)
    )


def test_no_route_shadows_an_imported_function() -> None:
    """No route handler may share a name with a function it imports (defect 1).

    ``async def compare_runs(...)`` in a module that also does
    ``from app.evaluation.regression import compare_runs`` binds the name in the
    module namespace. The body of the handler then resolves ``compare_runs(...)``
    to *itself*, passing two dicts to a function annotated
    ``-> RegressionReport``. The recursion bottoms out in a ``TypeError`` deep
    inside the loader, and the endpoint 500s with an error that names none of
    the things that actually went wrong.

    ``ruff check`` reports the same thing as F811, but only when the import is
    actually unused -- and here it was "used", by the very call it broke. This
    guard is about the hazard, not the linter.

    Limited to ``app.evaluation.*`` and ``app.services.*``: those are the
    function-bearing modules a route delegates to, and the ones where shadowing
    turns into infinite recursion. A route shadowing ``uuid`` or ``select``
    would be a different and much more visible bug.
    """
    offenders: list[tuple[str, int, str]] = []

    for path in _route_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (
                node.module.startswith("app.evaluation")
                or node.module.startswith("app.services")
            ):
                for alias in node.names:
                    imported[alias.asname or alias.name] = node.lineno
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name in imported:
                    offenders.append(
                        (
                            path.relative_to(APP_ROOT.parent).as_posix(),
                            node.name,
                            imported[node.name],
                        )
                    )

    assert not offenders, (
        "A route handler shadows a function it imports, so calling that function "
        "inside the handler recurses into the handler:\n"
        + "\n".join(
            f"  {file} {name}() shadows the import at line {import_line}"
            for file, name, import_line in offenders
        )
    )


# ---------------------------------------------------------------------------
# 2 + 4 + 5. Run-lifecycle helpers
# ---------------------------------------------------------------------------


def test_no_window_filter_is_fabricated_to_carry_an_organization() -> None:
    """No code builds a ``WindowFilter`` without a real time range (defect 2).

    ``WindowFilter`` requires ``start`` and ``end``. Several lookups -- "load
    this run by id", "load this experiment by id" -- have no time window at all
    and need only the organization id, and the natural-looking way to get it
    through an existing helper is to manufacture a filter with two invented
    datetimes. That raises ``TypeError: WindowFilter.__init__() missing 2
    required positional arguments`` at runtime, on every call, for every caller.

    The primitive for a windowless site is ``tenant_predicates``, which takes
    the ids directly. So this guard asserts that every ``WindowFilter(...)`` in
    the application supplies both fields.

    Note this cannot be a pure signature check: a ``WindowFilter`` built with
    keyword arguments puts the names in ``node.keywords``, while a positional one
    puts them in ``node.args``, and both are valid and both pass here.
    """
    offenders: list[tuple[str, int]] = []

    for path in sorted(APP_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "WindowFilter":
                continue
            supplied = {kw.arg for kw in node.keywords if kw.arg}
            supplied |= {"start", "end"} & {
                a.id for a in node.args if isinstance(a, ast.Name)
            }
            missing = {"start", "end"} - supplied
            if missing:
                offenders.append(
                    (path.relative_to(APP_ROOT.parent).as_posix(), node.lineno)
                )

    assert not offenders, (
        "A WindowFilter is built without a real time range. start/end are "
        "required fields, so this raises TypeError at runtime on every call. "
        "For a lookup with no window, use `tenant_predicates(organization_id, "
        "...)` from `app.repositories.common` instead:\n"
        + "\n".join(f"  {file}:{line}" for file, line in offenders)
    )


async def test_an_answer_run_aggregates_the_rows_it_stored(
    api_client: Any, db_session: Any
) -> None:
    """A persisted run's aggregate must describe the rows it actually kept (defect 5).

    ``SessionLocal`` sets ``autoflush=False``, so rows added earlier in the same
    transaction are invisible to a ``SELECT`` until something flushes them. The
    symptom is not a crash: the endpoint returns 200, the per-item rows commit
    correctly, and only the run's *stored* metrics read ``num_items: 0`` with
    every metric ``0.0``. It was found by comparing the run list against the
    run's own detail page, which is the only place the two disagree.

    So this asserts both halves against each other, and asserts the aggregate is
    not the all-zeros shape. A ``num_items`` of 0 with a real request body is the
    signature of the unflushed read; checking only that ``num_items > 0`` would
    pass on any batch whose true mean happened to be 0.0.
    """
    run_ids: list[uuid.UUID] = []
    try:
        response = await api_client.post(
            "/api/evaluations/answer",
            headers=PLATFORM_HEADERS,
            json={
                "name": f"flush-regression-{uuid.uuid4().hex[:8]}",
                "persist": True,
                "items": [
                    {
                        "question": "How do I reset my password?",
                        "answer": "Open Settings and choose Reset Password.",
                        "context": ["Open Settings and choose Reset Password."],
                    },
                    {
                        "question": "Where is the invoice history?",
                        "answer": "Invoices are listed under Billing.",
                        "context": ["Billing shows invoices and receipts."],
                    },
                ],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["num_items"] == 2, body

        # The run list, newest first. Identifying the run by its own stored
        # `num_items` rather than by name, because `EvaluationRunSummary` is
        # paged and a stale run from an earlier test could sit above this one --
        # and this assertion is specifically about the numbers this request wrote.
        runs = await api_client.get(
            "/api/evaluations/runs",
            headers=PLATFORM_HEADERS,
            params={"limit": 50, "evaluation_type": "answer"},
        )
        assert runs.status_code == 200, runs.text
        candidates = [
            r
            for r in runs.json()["items"]
            if (r.get("metrics") or {}).get("num_items") == 2
        ]
        assert candidates, (
            "No stored answer run reports num_items == 2. Stored: "
            f"{[(r['id'][:8], r.get('metrics')) for r in runs.json()['items']]}"
        )
        run_id = uuid.UUID(candidates[0]["id"])
        run_ids.append(run_id)

        # The stored aggregate, read back through the API rather than from the
        # ORM: this is the number the run-list page renders, so it is the one
        # that has to be right.
        detail = await api_client.get(
            f"/api/evaluations/runs/{run_id}", headers=PLATFORM_HEADERS
        )
        assert detail.status_code == 200, detail.text
        metrics = detail.json()["metrics"]

        assert metrics["num_items"] == 2, (
            "The run recorded a different item count than the response returned. "
            f"Stored {metrics} for a batch the endpoint reported as "
            f"num_items={body['num_items']}. With autoflush=False this means the "
            "aggregate was computed over rows that had not been flushed."
        )
        for field in (
            "faithfulness",
            "context_relevance",
            "answer_relevance",
            "citation_coverage",
            "unsupported_claim_ratio",
        ):
            assert metrics[field] == pytest.approx(body[field]), (
                f"{field}: stored {metrics[field]!r} but the endpoint returned "
                f"{body[field]!r} for the same batch"
            )
    finally:
        # Best-effort cleanup. The db_session fixture owns the transaction, so a
        # failure here must not mask the real assertion.
        for run_id in run_ids:
            try:
                await api_client.delete(
                    f"/api/evaluations/runs/{run_id}", headers=PLATFORM_HEADERS
                )
            except Exception:  # pragma: no cover - cleanup only
                pass


async def test_an_unjudged_run_records_no_judge_metrics(
    api_client: Any, db_session: Any
) -> None:
    """``use_llm_judge: false`` must not write a judge score of zero.

    The quarantine this file's module neighbours assert: a judge that did not run
    has no score, and the columns exist to say "not run". A stored
    ``judge_faithfulness: 0.0`` would be indistinguishable from a judge that ran
    and scored the batch at zero -- and, unlike ``None``, it would sort as the
    worst result on every dashboard that reads it.
    """
    response = await api_client.post(
        "/api/evaluations/answer",
        headers=PLATFORM_HEADERS,
        json={
            "name": f"judge-quarantine-{uuid.uuid4().hex[:8]}",
            "persist": False,
            "use_llm_judge": False,
            "items": [
                {
                    "question": "How do I reset my password?",
                    "answer": "Open Settings and choose Reset Password.",
                    "context": ["Open Settings and choose Reset Password."],
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["judge_faithfulness"] is None, body
    assert body["judge_answer_relevance"] is None, body
    assert body["judge_model"] is None, body
    # The deterministic side did run, and these are real numbers -- not None,
    # and not 0.0 either. That distinction is the whole point of the split.
    assert isinstance(body["faithfulness"], float)
    assert isinstance(body["context_relevance"], float)


def test_run_helpers_agree_on_an_identifier(api_client: Any) -> None:
    """``_open_run`` and ``_close_run`` must agree on what an id is (defect 4).

    ``_open_run`` returns the ORM row so the caller can read its freshly
    generated ``id``; ``_close_run`` takes that id and compares it to a column.
    Handing it the row makes SQLAlchemy try to inline a whole model as a SQL
    literal, and the endpoint 500s *at the end* of an otherwise successful
    batch -- every measurement already computed, every row already written, and
    the client still gets an error.

    Asserted against the source rather than by timing an endpoint failure,
    because the failure needs ``persist: true`` and a real embedder to
    reproduce, and a test that reproduces that to prove a type error is a slow
    test guarding a cheap invariant.
    """
    source = (APP_ROOT / "api" / "v1" / "answer_eval.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    close_param = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_close_run":
            close_param = node.args.args[1].arg if len(node.args.args) > 1 else None
    assert close_param == "run_id", (
        f"_close_run's second parameter is {close_param!r}, not 'run_id'. Its body "
        "compares that value against `EvaluationRun.id`, so anything that is not "
        "the identifier -- an ORM row, say -- is not legal as a SQL literal."
    )

    # Every call site must therefore pass `.id`, not the row. The session is the
    # first argument; the id is the second, and it is what the calls below pass.
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_close_run"
    ]
    assert calls, "_close_run is no longer called; this guard is looking in the wrong place"

    # The identifier's *provenance* is what matters, not the spelling at the
    # call site: `run_id` may legitimately be a local variable by then. So the
    # check is on the assignment that produced it -- anything an `_open_run(...)`
    # call flows into must have `.id` taken off it, or the ORM row is what
    # reaches `_close_run` one line later.
    sources: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "_open_run"
            for inner in ast.walk(node.value)
        ):
            takes_id = (
                isinstance(node.value, ast.Attribute) and node.value.attr == "id"
            )
            sources.append((node.lineno, "ok" if takes_id else ast.unparse(node.value)))

    assert sources, (
        "_open_run is no longer called; this guard is looking in the wrong place"
    )
    bad = [(line, value) for line, value in sources if value != "ok"]
    assert not bad, (
        f"The result of `_open_run(...)` is bound without taking `.id` at {bad}. "
        "`_close_run` compares its argument against `EvaluationRun.id`, so the "
        "row itself is not legal as a SQL literal and the endpoint 500s at the "
        "end of an otherwise successful batch."
    )
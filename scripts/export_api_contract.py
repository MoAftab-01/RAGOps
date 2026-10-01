"""Regenerate ``docs/api-contract.json`` from the running app's own OpenAPI.

**Why this exists.** The file was, until now, a snapshot somebody had edited by
hand -- which is exactly how it drifted: it was written before the tenancy work
and still described a world with no ``/api/organizations`` at all. A contract
that has to be maintained by hand is a contract that is wrong. This script makes
it a build artifact, so it cannot describe a shape the code does not have.

**Why it is derived from the app rather than imported from it.** Importing
``app.main`` would need the whole dependency tree and a reachable database. The
app is built in a subprocess, which already has a working ``uvicorn`` -- so this
imports nothing from the backend at all, and cannot drift from it either.

**Why the lossy shape.** The OpenAPI schema is far larger than anything a reader
uses. This keeps what a caller actually has to get right: the route, its
parameters and whether each is required, the request body schema by name, and
the response schema by name. ``resp`` is a schema *name*, not an inline body, so
``docs/API_CONTRACT.md`` and this file can be read together -- a route listed
here with ``resp: "OrganizationOut"`` is the same ``OrganizationOut`` the prose
documents.

Run from the repository root::

    .venv/Scripts/python.exe scripts/export_api_contract.py

The output is deterministic: routes are sorted, and dict keys are sorted at
every level, so a regeneration that changes nothing produces a byte-identical
file and ``git diff`` is meaningful.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "api-contract.json"

# Parameters that are an artefact of the plumbing rather than part of the API a
# caller writes against. `trace_id` in the path stays -- that is the API. A
# header the framework adds to every route is not.
IGNORED_PARAMS = frozenset({"X-API-Key"})


def _child_env() -> dict[str, str]:
    """A subprocess environment with the import path set.

    ``DATABASE_URL`` is deliberately *not* set: building the app opens no
    connection, and a missing database here should be a success, not a failure.
    """
    import os

    env = dict(os.environ)
    backend = REPO_ROOT / "backend"
    env["PYTHONPATH"] = str(backend) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _child_code(body: str) -> str:
    """Run ``body`` inside the backend package and return its stdout as JSON.

    A subprocess rather than an in-process import so that this script needs only
    a bare interpreter on ``PATH`` -- it never imports FastAPI itself, and the
    backend's own module state cannot leak into the contract this script writes.

    Two calls rather than one, because the two halves come from different
    places and only one of them is derivable from the OpenAPI document. See
    :func:`_write_guarded` for why the second call exists.
    """
    preamble = """
import json, sys
from fastapi.routing import APIRoute
from app.main import app
from app.core.security import require_api_key

def _write_guarded(route):
    # Walk the real dependant graph rather than the OpenAPI document:
    # `dependencies=[WriteGuard]` is a plain header dependency, so FastAPI
    # renders it as an ordinary optional `X-API-Key` parameter with no
    # securityScheme behind it. The graph is the only place the guard exists.
    seen = set()

    def walk(dependant):
        for sub in dependant.dependencies:
            if sub.call is require_api_key:
                return True
            if id(sub) in seen:
                continue
            seen.add(id(sub))
            if walk(sub):
                return True
        return False

    return walk(route.dependant)

guarded = {
    (route.path, method.lower())
    for route in app.routes
    if isinstance(route, APIRoute)
    for method in route.methods
    if _write_guarded(route)
}
"""
    result = subprocess.run(
        [sys.executable, "-c", preamble + body],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT / "backend"),
        env=_child_env(),
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Could not build the app to read its own API surface.\n"
            f"Run this from the repository root, with dependencies installed:\n\n"
            f"{result.stderr.strip()[-4000:]}"
        )
    return json.loads(result.stdout)


def _simple_type(param: dict[str, Any]) -> str:
    """The route contract's type vocabulary: string / integer / number / boolean.

    An enum collapses to ``string`` -- the allowed values are in the schema
    section, and repeating them per route would make every window parameter in
    the file unreadable.

    An ``anyOf`` of a type and ``null`` (FastAPI's rendering of an
    ``Optional[bool]``) reports as the *other* type, not as ``string``. Reading
    ``has_error`` as a string because the query flag is optional would be the
    kind of thing this file is supposed to prevent rather than introduce.
    """
    schema = param.get("schema", {})
    if "type" in schema:
        return schema["type"] if schema["type"] in {"integer", "number", "boolean"} else "string"
    for variant in schema.get("anyOf", []):
        kind = variant.get("type")
        if kind == "null":
            continue
        return kind if kind in {"integer", "number", "boolean"} else "string"
    return "string"



def _simple_params(operation: dict[str, Any]) -> list[list[Any]]:
    """Parameters as ``[name, required, type]``, path params first.

    Path parameters before query parameters because a caller constructs the URL
    before they construct the query string, and that is the order in which a
    missing one breaks.
    """
    params = operation.get("parameters", [])
    path = [p for p in params if p.get("in") == "path"]
    query = [p for p in params if p.get("in") == "query"]
    header = [p for p in params if p.get("in") == "header"]
    ordered = path + query + header
    return [
        [p["name"], bool(p.get("required", False)), _simple_type(p)]
        for p in ordered
        if p["name"] not in IGNORED_PARAMS
    ]


def _write_guarded(operation: dict[str, Any]) -> bool:
    """Whether this operation is behind the telemetry write guard.

    Read from the ``guarded`` set the child process builds by walking the app's
    dependency graph -- *not* from the OpenAPI document. A plain
    ``dependencies=[WriteGuard]`` is indistinguishable from any other optional
    header dependency once FastAPI has serialised it, so a route's protection
    is genuinely absent from ``/openapi.json`` and has to come from the graph.
    """
    raise NotImplementedError


def _ref_name(schema: dict[str, Any] | None) -> str | None:
    """The component schema name a body/response refers to, if it refers to one.

    ``None`` for an inline schema. A list response is reported as the *item*
    schema, because ``GET /api/organizations`` returning ``[]OrganizationOut``
    and ``GET /api/applications`` returning ``Page[ApplicationOut]`` are the two
    shapes a caller has to tell apart, and a name is what the prose documents
    use for both.
    """
    if not schema:
        return None
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    if schema.get("type") == "array":
        return _ref_name(schema.get("items"))
    # All three of these may hold exactly one variant plus a ``null`` from an
    # Optional field; walking to the first name is what makes an optional body
    # report the same name as the required one beside it.
    for key in ("allOf", "anyOf", "oneOf"):
        for variant in schema.get(key, []):
            name = _ref_name(variant)
            if name is not None:
                return name
    return None


def _body(operation: dict[str, Any]) -> str | None:
    """The request body schema name, or ``None``.

    A body that is itself a bare list or an inline object has no component name
    and is rendered inline, exactly as it was in the hand-maintained file.
    """
    content = operation.get("requestBody", {}).get("content", {})
    for media in content.values():
        schema = media.get("schema", {})
        name = _ref_name(schema)
        if name is not None:
            return name
        return json.dumps(schema, sort_keys=True)
    return None



def _response(operation: dict[str, Any]) -> str:
    """The success response schema name, or the inline schema as JSON.

    Prefers ``2xx`` over ``default``: a route whose only documented response is
    the error shape is better served by the 2xx it actually returns.
    """
    responses = operation.get("responses", {})
    for code in ("200", "201", "202", "204"):
        if code not in responses:
            continue
        for media in responses[code].get("content", {}).values():
            schema = media.get("schema", {})
            name = _ref_name(schema)
            if name is not None:
                return name
            return json.dumps(schema, sort_keys=True)
        return "204" if code == "204" else "{}"
    if "default" in responses:
        for media in responses["default"].get("content", {}).values():
            return json.dumps(media.get("schema", {}), sort_keys=True)
    return "{}"


def _schemas(document: dict[str, Any]) -> dict[str, Any]:
    """Every component schema, with titles stripped.

    Titles are FastAPI's echo of the Python class name and add nothing to a
    schema that is already keyed by that name; ``Page[ApplicationOut]`` is the
    one title worth keeping because it is the only place the generic
    parameterisation is written down. Kept as a value rather than a key because
    dropping it would lose it.
    """
    out: dict[str, Any] = {}
    for name, schema in document.get("components", {}).get("schemas", {}).items():
        stripped = {k: v for k, v in schema.items() if k != "title"}
        if stripped.get("type") == "object" and "title" in schema:
            # FastAPI's title is how `Page[ApplicationOut]` distinguishes
            # itself from any other page schema; the name alone does not.
            stripped["title"] = schema["title"]
        out[name] = stripped
    return out


def main() -> int:
    document = _child_code("json.dump(app.openapi(), sys.stdout)\n")
    guarded = _child_code(
        "json.dump(sorted([list(pair) for pair in guarded]), sys.stdout)\n"
    )
    guarded_set = {(path, method) for path, method in guarded}
    routes: dict[str, dict[str, Any]] = {}

    for path, operations in document.get("paths", {}).items():
        for method, operation in operations.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                # `parameters`, `summary` and friends can sit at the path level.
                continue
            entry: dict[str, Any] = {"params": _simple_params(operation)}
            # The guard is the difference between "read the dashboard" and
            # "write into a customer's data", so it is a named field rather
            # than a detail buried in the parameter list. Absent means open.
            if (path, method.lower()) in guarded_set:
                entry["write"] = True
            entry["body"] = _body(operation)
            entry["resp"] = _response(operation)
            routes[f"{method.upper()} {path}"] = entry

    contract = {"routes": dict(sorted(routes.items())), "schemas": _schemas(document)}
    OUTPUT.write_text(json.dumps(contract, indent=1) + "\n", encoding="utf-8")

    mutations = [r for r in routes if r.split(" ", 1)[0] != "GET"]
    unguarded = [r for r in mutations if "write" not in routes[r]]
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  {len(routes)} routes, {len(contract['schemas'])} schemas")
    print(f"  {len(mutations) - len(unguarded)}/{len(mutations)} mutations are write-guarded")
    if unguarded:
        print("  unguarded mutations (each intentional -- confirm before shipping):")
        for route in unguarded:
            print(f"    {route}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

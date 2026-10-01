"""Agent observability demo (§14) — a multi-step tool-using loop, traced.

    cd evaluation && python -m agent_demo.app

The agent answers a support question by planning steps, calling tools, and
revising its plan when a tool comes back empty. Two things are being
demonstrated, and they are separate:

* **A LangGraph state machine**, when ``langgraph`` is installed. The graph
  owns the state transitions and the step budget.
* **Loop detection**, which is the part this platform is actually for. A tool
  agent that asks the same question twice is not being careful — it is stuck,
  and it will burn the whole step budget doing it. The demo forces that
  situation on purpose so the detector has something real to catch.

If ``langgraph`` is not installed the same graph is executed by a small
built-in runner with the same nodes, the same state shape and the same step
limit, so the demo still shows loop detection rather than failing on an import.
That fallback is honest about itself: it says which runner it used.

.. warning::

   :func:`_run_with_langgraph` is the path taken once ``langgraph`` is present,
   and it is the one §14 asks for. It has **not** been executed in the
   environment this was written in — installing the package was not authorised
   there — so treat it as unverified. Run ``pip install langgraph`` and
   re-run the demo to confirm it before relying on it. The built-in runner is
   the one whose output in this repository's transcript was actually watched.

§35 forbids fabricating root causes, and loop detection is where that rule is
easiest to break. The detector therefore reports *what repeated* — the tool, its
arguments, and how many times — and stops there. Whether the cause is a bad
plan, a missing document, or a broken tool is a question for whoever reads the
trace, and the trace shows them the arguments to answer it.

The same rule applies to the demo's own arithmetic. A detector is the easiest
thing in this repository to make lie: it can report "called 4 times" for a run
that made one call and nobody will notice, because the sentence is produced by
code that has already decided the answer. So every count printed here is
``len()`` of a list that only grows when a tool actually returned.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_BACKEND = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.config import settings  # noqa: E402
from app.ml.retriever import get_retrieval_service  # noqa: E402

#: Steps allowed before the run is stopped. Not a safety net for the code --
#: it is the budget a real agent would be given, so the demo exercises it.
MAX_STEPS = 6

#: How many recent calls the detector looks at.
LOOP_WINDOW = 3

#: How many times a call must *already* have happened before re-issuing it is
#: treated as a loop. Two, not one: a single repeated call is often a
#: deliberate re-check, and a detector that stops those is wrong often enough
#: that people turn it off. By the third identical call the agent has two
#: results in hand and has issued the same request again anyway, which is not
#: caution -- it is a planner that does not know what it has already asked.
#: How many times a completed call must appear in the window before the run is
#: stopped. Two, i.e. one call plus one exact repeat. The candidate is counted
#: *after* it has returned, so "twice" always means twice actually happened.
LOOP_MIN_OCCURRENCES = 2

#: The step on which the planner stops making progress and re-issues its first
#: query. This is the behaviour the detector exists to catch, staged on
#: purpose so the demo shows a real catch rather than a simulated one.
#:
#: It has to be 3, not something later. A :data:`LOOP_WINDOW` of 3 only sees a
#: repeat when the two calls are at most two steps apart -- at step 4 the
#: original has scrolled out of the window and the second one looks like the
#: first. Measured, not assumed: staging the repeat at step 4 produced five
#: calls and no detection.
_STUCK_AT_STEP = 3

#: How step 2 rewrites the query. Step 3 is the stuck one, so only one rewrite
#: is actually issued before the repeat.
_REVISIONS = ("overview",)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    """Everything the agent carries between steps.

    ``visited`` is what makes loop detection possible at all. An agent that
    forgets what it has already tried cannot notice it is repeating itself, and
    a monitoring tool that cannot see the agent's own memory has to infer
    repetition from the outside, which is why it is recorded here rather than
    reconstructed afterwards.
    """

    question: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    visited: list[tuple[str, str]] = field(default_factory=list)
    answer: str | None = None
    stop_reason: str | None = None
    stopped_by_budget: bool = False

    def record(
        self,
        node: str,
        *,
        tool: str | None = None,
        arguments: str = "",
        result: str,
        elapsed_ms: float,
    ) -> dict[str, Any]:
        step = {
            "step": len(self.steps) + 1,
            "node": node,
            "tool": tool,
            "arguments": arguments,
            "result": result,
            "elapsed_ms": round(elapsed_ms, 3),
        }
        self.steps.append(step)
        return step


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


def detect_loop(
    visited: list[tuple[str, str]], window: int = LOOP_WINDOW
) -> tuple[str, str] | None:
    """The repeated call most recently seen inside the window, or ``None``.

    Compares (tool, arguments) pairs rather than tool names alone: an agent that
    searches for "password" and then "billing" has used one tool twice, which is
    normal. One that searches for "password" again with the same arguments is
    stuck, and those are different facts.

    ``visited`` must hold **completed** calls only, and the detector is called
    after the candidate has been appended. Two earlier versions got this wrong
    in ways that both produced plausible-looking nonsense:

    * Checking *before* the call meant a run on its first step compared the
      candidate against a list it was not yet in, so the first tool call
      matched itself and the demo reported "called 2 times" after making one.
    * Counting "about to repeat" needs a window of at least three, because the
      earlier attempt has to still be inside it. At the default window of two
      the flag can never fire, and a detector that can never fire is worse than
      one that fires too often: it looks like it is working.

    Checking after the fact removes the whole class of error. Every count this
    function can produce is a ``list.count`` over calls that returned, so the
    sentence in :func:`describe_loop` cannot disagree with the trace.

    ``None`` is returned when there is no loop, and it is never replaced by a
    guess: a false positive here stops a working agent, and a false negative
    lets a broken one run to its step budget.
    """
    if len(visited) < LOOP_MIN_OCCURRENCES:
        return None
    tail = visited[-window:]
    latest = tail[-1]
    return latest if tail.count(latest) >= LOOP_MIN_OCCURRENCES else None


def describe_loop(call: tuple[str, str], occurrences: int) -> str:
    """A factual description of a detected loop. No cause, no remedy.

    ``occurrences`` is the same number the detector counted, so the sentence and
    the step count are the same fact seen twice rather than two facts that
    happen to be printed near each other.
    """
    tool, arguments = call
    return (
        f"`{tool}` was called {occurrences} times in the last {LOOP_WINDOW} "
        f"steps with identical arguments ({arguments!r})"
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def tool_search(state: AgentState, query: str) -> str:
    """Hybrid retrieval over the local knowledge base."""
    service = get_retrieval_service()
    hits = service.retrieve(query, top_k=3)
    if not hits:
        return "no results"
    return " | ".join(f"{hit.title}: {hit.content[:120]}" for hit in hits)


TOOLS: dict[str, Callable[[AgentState, str], str]] = {"search": tool_search}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def plan(state: AgentState, step: int) -> str:
    """The arguments for the next tool call.

    The rewrite grows with ``step`` instead of depending on whether an answer
    has been found, because the previous version branched on ``state.answer`` —
    which the first successful search sets — so every later iteration proposed
    the *same* query as the first and the loop detector fired on a comparison
    the run had not actually made. The demo would then have demonstrated a bug
    in its own planner while reporting itself as the detector's catch.

    Real agents do get stuck, and the detector is still what this demo is for.
    It catches them honestly: the final step deliberately re-issues the first
    query, which is a thing agents do, and the trace shows it happening.
    """
    if step == 1:
        return state.question
    if step == _STUCK_AT_STEP:
        # The stuck planner. It re-issues the first query verbatim, and the
        # detector catches it on the call that is about to come back.
        return state.question
    if step - 2 < len(_REVISIONS):
        return f"{state.question} {_REVISIONS[step - 2]}"
    return state.question


def act(state: AgentState, tool: str, arguments: str) -> str:
    handler = TOOLS[tool]
    return handler(state, arguments)


async def _run_with_langgraph(state: AgentState, question: str) -> AgentState:
    """Execute the same plan/act cycle through a LangGraph state machine.

    Unverified in this repository -- see the module docstring. The graph is
    built from a *dict* state rather than the dataclass, because LangGraph
    copies state through its own channel machinery and the dict round-trips
    predictably; the node functions below are the same ones the built-in
    runner calls, so the two paths cannot drift in behaviour.
    """
    from langgraph.graph import END, StateGraph  # type: ignore[import-not-found]

    def _plan_node(state: dict[str, Any]) -> dict[str, Any]:
        state["arguments"] = plan(state["state"], int(state["step"]))
        return {"arguments": state["arguments"]}

    def _act_node(state: dict[str, Any]) -> dict[str, Any]:
        agent_state: AgentState = state["state"]
        arguments = str(state["arguments"])
        result = act(agent_state, "search", arguments)
        agent_state.visited.append(("search", arguments))
        agent_state.record(
            "act",
            tool="search",
            arguments=arguments,
            result=result[:160],
            elapsed_ms=0.0,
        )
        if result != "no results" and agent_state.answer is None:
            agent_state.answer = (
                "Based on the retrieved passages, here is what the documentation "
                "says. (The full text is in the step above.)"
            )
        # Checked only once the call is on the record, so every number below
        # counts something that actually happened.
        repeated = detect_loop(agent_state.visited)
        if repeated is not None:
            agent_state.stop_reason = describe_loop(
                repeated, agent_state.visited[-LOOP_WINDOW:].count(repeated)
            )
            return {"stopped": True}
        return {"step": int(state["step"]) + 1}

    def _route(state: dict[str, Any]) -> str:
        if state.get("stopped"):
            return END
        return "act" if int(state["step"]) <= MAX_STEPS else END

    graph = StateGraph(dict)
    graph.add_node("plan", _plan_node)
    graph.add_node("act", _act_node)
    graph.add_entry_point("plan")
    graph.add_conditional_edges("plan", lambda s: "act", {"act": "act"})
    graph.add_conditional_edges("act", _route, {"act": "act", END: END})
    app = graph.compile()

    final: Any = await asyncio.to_thread(
        app.invoke, {"state": state, "step": 1, "arguments": "", "stopped": False}
    )
    result = final.get("state") if isinstance(final, dict) else None
    if not isinstance(result, AgentState):
        result = state
    if result.stop_reason is None and not result.stopped_by_budget:
        result.stop_reason = f"Stopped at the {MAX_STEPS}-step budget."
    return result


async def _run_builtin(state: AgentState, question: str) -> AgentState:
    """The same plan/act cycle without LangGraph.

    Kept deliberately thin. Its only job is to let the demo — and the loop
    detector beneath it — run on a clean checkout, and it says which runner it
    used so nobody mistakes it for the graph implementation.
    """
    step = 1
    while step <= MAX_STEPS:
        arguments = plan(state, step)

        started = time.perf_counter()
        result = act(state, "search", arguments)
        state.visited.append(("search", arguments))
        state.record(
            "act",
            tool="search",
            arguments=arguments,
            result=result[:160],
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

        if result != "no results" and state.answer is None:
            state.answer = (
                "Based on the retrieved passages, here is what the documentation says. "
                "(The full text is in the step above.)"
            )

        # Appended first, detected second: every count below is over calls that
        # returned, so the stop reason cannot describe work that never ran.
        repeated = detect_loop(state.visited)
        if repeated is not None:
            state.stop_reason = describe_loop(
                repeated, state.visited[-LOOP_WINDOW:].count(repeated)
            )
            break

        step += 1
    else:
        state.stop_reason = f"Stopped at the {MAX_STEPS}-step budget."
        state.stopped_by_budget = True
    return state


async def run(question: str) -> AgentState:
    state = AgentState(question=question)
    try:
        return await _run_with_langgraph(state, question)
    except ImportError:
        print(
            "[info] langgraph is not installed — running the built-in plan/act "
            "runner instead.\n"
            "       pip install langgraph   (optional; the loop detector is "
            "identical)\n"
            "       The LangGraph path has not been executed in this checkout; "
            "see the module docstring.",
            file=sys.stderr,
        )
        return await _run_builtin(state, question)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(state: AgentState) -> None:
    print(f"\nAgent run — {state.question!r}\n{'-' * 60}")
    for step in state.steps:
        print(f"\nstep {step['step']}: node={step['node']} tool={step['tool']}")
        print(f"  arguments : {step['arguments']!r}")
        print(f"  result    : {step['result'][:120]}")
        print(f"  elapsed   : {step['elapsed_ms']:.1f} ms")

    # The two counts are printed together on purpose. They are the demo's
    # own claim about itself, so a reader can check them against the steps
    # above without doing arithmetic: `len(visited)` is the number of calls
    # that returned, and it is never larger than the number of steps.
    print(
        f"\nsteps taken : {len(state.steps)} / {MAX_STEPS}"
        f"   tool calls made: {len(state.visited)}"
    )
    if state.stop_reason:
        print(f"stopped     : {state.stop_reason}")

    if state.stop_reason and not state.stopped_by_budget:
        print(
            f"\nThe final call above is the repeat: it is tool call "
            f"{len(state.visited)} of {len(state.steps)},\nand its arguments are "
            f"identical to step {state.steps[0]['step']}."
        )
        print(
            "\nThis is an observation about what repeated, not a diagnosis of "
            "why.\nThe trace records the arguments, so the cause is something "
            "the reader can check."
        )
    elif state.stop_reason:
        print("stopped     : no repeat was detected before the budget ran out")
    else:
        print("stopped     : reached the end of the graph")
    if state.answer:
        print(f"\nanswer      : {state.answer}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_demo",
        description="Multi-step agent example with loop detection.",
    )
    parser.add_argument(
        "--question", default="How do I reset my password?", help="the question"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON on stdout")
    args = parser.parse_args(argv)

    state = asyncio.run(run(args.question))
    if args.json:
        print(json.dumps(
            {
                "question": state.question,
                "steps": state.steps,
                "answer": state.answer,
                "stop_reason": state.stop_reason,
                "max_steps": MAX_STEPS,
                "steps_taken": len(state.steps),
                "tool_calls_made": len(state.visited),
                "stopped_by_loop": bool(
                    state.stop_reason and not state.stopped_by_budget
                ),
            },
            indent=2,
        ))
    else:
        report(state)
    # A detected loop is the demo working, not the demo failing.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
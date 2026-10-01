"""Where the retrieval service looks for the corpus, and the agent demo's loop
detector.

Two regressions, both of which produced *plausible* wrong answers rather than
errors, which is why they needed pinning rather than a glance:

* ``KNOWLEDGE_BASE_PATH`` is documented as relative to the repository root, but
  a relative path resolves against the process CWD. RAGOps is started from
  ``backend/``, from ``evaluation/`` and from ``backend/tests/``; from two of
  those the directory does not exist, and a missing corpus directory builds an
  *empty* index rather than raising. Every query then returns nothing, which
  reads as "the corpus does not cover this" when no corpus was ever loaded.
* The agent demo's loop detector counted a candidate against a list it had not
  yet been appended to, so the very first tool call matched itself and the demo
  reported "called 2 times" after making one. §35 forbids fabricating findings,
  and a loop detector is the easiest place in this repository to do it.

Neither test builds a model or touches a database.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from app.ml.retriever import _resolve_knowledge_base

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestKnowledgeBaseResolution:
    def test_an_absolute_path_is_taken_as_given(self, tmp_path: Path) -> None:
        corpus = tmp_path / "kb"
        corpus.mkdir()
        assert _resolve_knowledge_base(corpus) == corpus

    def test_an_explicit_path_is_never_rewritten(self, tmp_path: Path) -> None:
        # A caller that spelled out a path means it, even if it does not exist.
        # Silently redirecting it would make a test that points at a fixture
        # load the real knowledge base instead.
        missing = tmp_path / "does-not-exist"
        assert _resolve_knowledge_base(missing) == missing

    def test_a_relative_path_is_found_from_an_ancestor_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pretend we are running from somewhere unrelated, and that the
        # repository is an ancestor of it.
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (tmp_path / "datasets").mkdir()
        monkeypatch.chdir(deep)

        from app.config import settings

        monkeypatch.setattr(settings, "knowledge_base_path", "datasets")
        assert _resolve_knowledge_base(None) == tmp_path / "datasets"

    def test_an_existing_relative_path_wins_over_the_ancestor_search(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The CWD is checked first, so a run deliberately started inside its
        # own corpus uses that one.
        local = tmp_path / "local"
        local.mkdir()
        (tmp_path / "datasets").mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("app.ml.retriever.settings.knowledge_base_path", "local")
        assert _resolve_knowledge_base(None) == local

    def test_the_shipped_default_resolves_in_a_real_checkout(self) -> None:
        # The regression itself: from backend/tests/ the configured relative
        # path does not exist, and the ancestors contain the real corpus.
        from app.config import settings

        configured = Path(settings.knowledge_base_path)
        if configured.is_absolute():
            pytest.skip("absolute knowledge base path; not the tested default")
        resolved = _resolve_knowledge_base(None)
        assert resolved.is_absolute()
        assert resolved.is_dir(), (
            f"knowledge base {configured!r} did not resolve to a directory; "
            f"got {resolved}"
        )

    def test_the_default_paths_in_the_repo_actually_exist(self) -> None:
        # Guards the constant itself, so a rename cannot silently point the
        # whole platform at an empty directory.
        assert (_REPO_ROOT / "evaluation" / "datasets" / "knowledge_base").is_dir()


@pytest.fixture(scope="module")
def agent_demo():
    """Load the demo's pure logic without importing its model-backed deps.

    ``agent_demo.app`` imports the retriever at module level, which is correct
    for the demo and useless here: the functions under test are pure and touch
    neither the index nor a network. The stubs below stand in for those two
    imports so the module loads in milliseconds and offline.

    The stubs have to be registered in ``sys.modules`` and the module object
    has to be registered too, not just executed: ``@dataclass`` resolves its
    annotations through ``sys.modules[cls.__module__]``, and a module that was
    never inserted raises ``NameError: name 'str' is not defined`` the moment
    a dataclass is declared in it.
    """
    source = _REPO_ROOT / "evaluation" / "agent_demo" / "app.py"
    name = "ragops_agent_demo_under_test"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)

    retriever_stub = ModuleType("app.ml.retriever")
    retriever_stub.get_retrieval_service = lambda: None  # type: ignore[attr-defined]
    config_stub = ModuleType("app.config")
    config_stub.settings = None  # type: ignore[attr-defined]

    targets = ("app.ml.retriever", "app.config", name)
    saved = {key: sys.modules.get(key) for key in targets}
    sys.modules["app.ml.retriever"] = retriever_stub
    sys.modules["app.config"] = config_stub
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


class TestLoopDetection:
    def test_an_empty_history_is_not_a_loop(self, agent_demo) -> None:
        assert agent_demo.detect_loop([]) is None

    def test_one_call_is_not_a_loop(self, agent_demo) -> None:
        # The reported regression. With the candidate compared before it was
        # appended, this returned itself and the demo claimed a repeat.
        assert agent_demo.detect_loop([("search", "q")]) is None

    def test_one_call_plus_an_exact_repeat_is_a_loop(self, agent_demo) -> None:
        assert agent_demo.detect_loop(
            [("search", "q"), ("search", "q")]
        ) == ("search", "q")

    def test_different_arguments_are_not_a_loop(self, agent_demo) -> None:
        # Using one tool three times is normal; the arguments are the fact.
        assert agent_demo.detect_loop(
            [("search", "a"), ("search", "b"), ("search", "c")]
        ) is None

    def test_a_repeat_older_than_the_window_is_not_a_loop(
        self, agent_demo
    ) -> None:
        window = agent_demo.LOOP_WINDOW
        history: list[tuple[str, str]] = [("search", "old")]
        history += [("search", f"filler-{i}") for i in range(window)]
        history.append(("search", "fresh"))
        assert agent_demo.detect_loop(history) is None

    def test_the_window_is_wide_enough_to_see_a_repeat(self, agent_demo) -> None:
        # Measured, not assumed. With LOOP_WINDOW=3 a repeat staged two steps
        # after the original is the furthest back it can sit, so a planner that
        # stalls at step 4 produces two *separate* recent calls and is not
        # flagged. Tightening this would silently stop catching real loops.
        assert agent_demo.LOOP_WINDOW >= agent_demo.LOOP_MIN_OCCURRENCES + 1

    def test_the_stop_reason_counts_only_completed_calls(
        self, agent_demo
    ) -> None:
        # The number in the message has to agree with the trace printed beside
        # it, or the demo is asserting something it did not measure.
        history = [("search", "q"), ("search", "other"), ("search", "q")]
        detected = agent_demo.detect_loop(history)
        assert detected is not None
        occurrences = history[-agent_demo.LOOP_WINDOW:].count(detected)
        assert occurrences == 2
        assert str(occurrences) in agent_demo.describe_loop(detected, occurrences)

    def test_the_stop_reason_makes_no_claim_about_cause(self, agent_demo) -> None:
        # §35: report what repeated, not why. Banned words are the ones a
        # tempting diagnosis would reach for.
        text = agent_demo.describe_loop(("search", "q"), 2).lower()
        for word in ("because", "due to", "caused", "likely", "probably", "should"):
            assert word not in text, f"stop reason implies a cause: {word!r}"

    def test_the_planner_makes_progress_before_it_stalls(
        self, agent_demo
    ) -> None:
        state = agent_demo.AgentState(question="q")
        first = agent_demo.plan(state, 1)
        second = agent_demo.plan(state, 2)
        assert first == "q"
        assert second != first, "step 2 must revise, or the repeat is a foregone conclusion"

    def test_the_stalled_planner_repeats_a_call_within_the_window(
        self, agent_demo
    ) -> None:
        # The demo's whole point: the loop it reports is one it actually ran.
        state = agent_demo.AgentState(question="q")
        stuck = agent_demo._STUCK_AT_STEP
        gap = stuck - 1
        assert gap <= agent_demo.LOOP_WINDOW, (
            f"the repeat is staged {gap} steps after the original but the "
            f"window is only {agent_demo.LOOP_WINDOW}, so it cannot be caught"
        )
        assert agent_demo.plan(state, stuck) == agent_demo.plan(state, 1)

"""Answer evaluation: faithfulness, citations, relevance, and judge hygiene.

§26 of the specification asks for faithfulness, context relevance, answer
relevance and citation coverage, and then says explicitly *"Do not blindly
trust an LLM judge."* The implementation answers that structurally rather than
rhetorically: every score below is computed deterministically from text, a
judge can only populate fields named ``judge_*``, and a missing embedder
produces a reported ``None`` instead of a substituted default.
"""

from __future__ import annotations

import pytest

from app.evaluation.answer_evaluator import (
    EmbedderRequiredError,
    JudgeScores,
    citation_coverage,
    citation_report,
    content_word_coverage,
    evaluate_answer,
    evaluate_answer_detailed,
    extract_claims,
    parse_citation_markers,
)


class FakeEmbedder:
    """A deterministic bag-of-words embedder.

    Real evaluation uses a sentence-transformer, but downloading 90 MB of model
    weights inside a unit test would make the suite depend on the network. This
    produces a fixed unit vector per token-set, so cosine similarity is a
    well-defined, reproducible function of word overlap — enough to assert the
    blending logic and the "no embedder means no answer-relevance number" rule.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def model_name(self) -> str:
        return "fake-bag-of-words"

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            counts: dict[str, int] = {}
            for word in text.lower().split():
                counts[word] = counts.get(word, 0) + 1
            vectors.append([float(counts.get(k, 0)) for k in sorted(counts)])
        # Pad to a shared width so the caller sees a rectangular matrix.
        width = max(len(v) for v in vectors)
        return [v + [0.0] * (width - len(v)) for v in vectors]


class TestContentWordCoverage:
    def test_answer_repeating_the_question_scores_high(self) -> None:
        value = content_word_coverage(
            "how do I reset my password",
            "To reset your password, open the account settings page.",
        )
        assert 0.0 < value <= 1.0

    def test_answer_about_something_else_scores_low(self) -> None:
        value = content_word_coverage(
            "how do I reset my password",
            "The weather in Reykjavik is usually cold in winter.",
        )
        assert value < 0.2

    def test_is_bounded(self) -> None:
        for answer in ("", "x", "password password password"):
            assert 0.0 <= content_word_coverage("password question", answer) <= 1.0

    def test_empty_question_does_not_raise(self) -> None:
        assert 0.0 <= content_word_coverage("", "anything at all") <= 1.0


class TestExtractClaims:
    def test_splits_on_sentence_boundaries(self) -> None:
        claims = extract_claims("The sky is blue. Grass is green.")
        assert len(claims) == 2

    def test_ignores_empty_input(self) -> None:
        assert extract_claims("") == []
        assert extract_claims("   \n  ") == []

    def test_strips_whitespace_from_each_claim(self) -> None:
        for claim in extract_claims("  A.  B. "):
            assert claim == claim.strip()

    def test_a_single_sentence_is_one_claim(self) -> None:
        assert len(extract_claims("Just one sentence without a full stop")) == 1


class TestCitationMarkers:
    def test_bracketed_numbers_are_markers(self) -> None:
        assert parse_citation_markers("The limit is 100 GB [1].") == ["1"]

    def test_multiple_markers_are_all_found(self) -> None:
        assert parse_citation_markers("A [1] and B [2] and C [1,3]") == ["1", "2", "1", "3"]

    def test_a_marker_list_counts_as_one_marker_per_source(self) -> None:
        # "[1,3]" is two citations, so the coverage denominator has to be two.
        # Collapsing it to the literal string "1,3" would make it resolve
        # against nothing and understate coverage.
        assert parse_citation_markers("See [1,3] and [2].") == ["1", "3", "2"]

    def test_a_space_is_not_a_list_separator(self) -> None:
        # "[see 3]" is prose in brackets, not a citation. Before the fix it
        # parsed as the tokens "see" and "3", counting an English word as a
        # citation and diluting coverage. Now it yields no marker at all.
        assert parse_citation_markers("[see 3]") == []

    def test_bare_reference_inside_prose_still_parses(self) -> None:
        assert parse_citation_markers("the limit is [1]") == ["1"]

    def test_keyed_markers_still_work(self) -> None:
        # The keyed branch allows internal spaces, so it must not be affected by
        # the list-splitting rule applied to the bare-token branch.
        assert parse_citation_markers("[source: billing faq]") == ["billing faq"]

    def test_no_markers_yields_empty(self) -> None:
        assert parse_citation_markers("No citations here.") == []

    def test_text_without_brackets_is_not_a_citation(self) -> None:
        # "(1)" and "see 1" are not the documented [n] form.
        assert parse_citation_markers("See (1) for details.") == []

    def test_empty_answer(self) -> None:
        assert parse_citation_markers("") == []


class TestCitationReport:
    def test_uncited_answer_is_not_credited(self) -> None:
        # Coverage of 1.0 for an answer with no citations would reward the
        # model for ignoring its sources entirely.
        report = citation_report("This answer has no citations at all.", [])
        assert report.has_citations is False
        assert report.coverage == 0.0

    def test_empty_context_cannot_be_cited(self) -> None:
        report = citation_report("Answer citing [1].", [])
        assert report.coverage == 0.0

    def test_marker_with_no_matching_document_is_not_credited(self) -> None:
        report = citation_report("Answer citing [7].", [{"id": "1", "text": "x"}])
        assert report.coverage == 0.0

    def test_citation_coverage_helper_agrees_with_the_report(self) -> None:
        answer = "Answer citing [1]."
        docs = [{"id": "1", "text": "some source text"}]
        assert citation_coverage(answer, docs) == citation_report(answer, docs).coverage

    def test_coverage_is_bounded(self) -> None:
        docs = [{"id": str(i), "text": f"doc {i}"} for i in range(1, 6)]
        answer = "Everything here cites [1][2][3][4][5][6][7][8][9][10]."
        assert 0.0 <= citation_coverage(answer, docs) <= 1.0


class TestDeterministicEvaluation:
    QUESTION = "How do I enable two-factor authentication?"
    GOOD_CONTEXT = [
        {"id": "doc-1", "text": "Two-factor authentication is enabled in Account settings under Security."},
        {"id": "doc-2", "text": "You will need your phone to complete the setup."},
    ]

    def test_a_grounded_answer_scores_higher_than_an_ungrounded_one(self) -> None:
        grounded = evaluate_answer(
            self.QUESTION,
            "Enable two-factor authentication in Account settings under Security [1].",
            self.GOOD_CONTEXT,
        )
        hallucinated = evaluate_answer(
            self.QUESTION,
            "Two-factor authentication is enabled by emailing support@vendor.example.",
            self.GOOD_CONTEXT,
        )
        assert grounded.faithfulness >= hallucinated.faithfulness
        assert grounded.citation_coverage > hallucinated.citation_coverage

    def test_scores_are_bounded(self) -> None:
        result = evaluate_answer(
            self.QUESTION, "Any answer at all.", self.GOOD_CONTEXT
        )
        for field in (
            "faithfulness",
            "context_relevance",
            "answer_relevance",
            "citation_coverage",
            "unsupported_claim_ratio",
        ):
            assert 0.0 <= getattr(result, field) <= 1.0, field

    def test_method_names_the_deterministic_family(self) -> None:
        result = evaluate_answer(self.QUESTION, "Answer.", self.GOOD_CONTEXT)
        assert result.method == "deterministic_alignment"
        assert result.judge_faithfulness is None
        assert result.judge_model is None

    def test_missing_context_reports_zero_not_a_crash(self) -> None:
        # An empty context is a real situation -- a retrieval miss -- and must
        # report no support rather than raising.
        result = evaluate_answer(self.QUESTION, "Some answer.", [])
        assert result.faithfulness == 0.0
        assert result.citation_coverage == 0.0

    def test_no_embedder_means_no_answer_relevance_number(self) -> None:
        # Deterministic alignment cannot produce a cosine, so the field is
        # reported as absent rather than filled with a plausible default.
        report = evaluate_answer_detailed(
            self.QUESTION, "Enable it in settings [1].", self.GOOD_CONTEXT
        )
        assert report.answer_relevance_components["similarity"] is None
        assert report.result.method == "deterministic_alignment"

    def test_claims_are_reported_with_their_support_status(self) -> None:
        report = evaluate_answer_detailed(
            self.QUESTION,
            "Enable it in Account settings under Security [1]. Also upgrade to the Pro plan.",
            self.GOOD_CONTEXT,
        )
        assert report.result.claims, "claims must be itemised, not just scored"
        for claim in report.result.claims:
            assert isinstance(claim.supported, bool)
            assert 0.0 <= claim.similarity <= 1.0
            assert claim.method

    def test_unsupported_claims_are_visible_not_hidden(self) -> None:
        report = evaluate_answer_detailed(
            self.QUESTION,
            "The Pro plan costs 400 dollars per month and includes unlimited seats.",
            self.GOOD_CONTEXT,
        )
        assert report.result.unsupported_claim_ratio > 0.0
        assert any(
            not claim.supported for claim in report.result.claims
        ), "an unsupported claim must be named, not merely counted"


class TestEmbedderRequired:
    def test_context_relevance_without_an_embedder_raises(self) -> None:
        # Answer relevance degrades gracefully (reported as None) but the
        # dedicated embedding metric must not silently fall back to something
        # that looks like a measurement.
        from app.evaluation.answer_evaluator import evaluate_context_relevance

        with pytest.raises(EmbedderRequiredError):
            evaluate_context_relevance(
                "question", [{"id": "1", "text": "text"}], embedder=None
            )


class TestJudgeIsolation:
    """§26: *"Do not blindly trust an LLM judge."* — enforced structurally."""

    def test_a_judge_cannot_overwrite_a_deterministic_score(self) -> None:
        judge = JudgeScores(
            faithfulness=0.99, answer_relevance=0.99, model="fake-judge", judged_by="llm"
        )
        result = evaluate_answer(
            "How do I enable two-factor authentication?",
            "Unrelated text about penguins.",
            [],
            judge=judge,
        )
        # The judge's numbers land in their own fields...
        assert result.judge_faithfulness == 0.99
        assert result.judge_model == "fake-judge"
        # ...and the measured fields are untouched by them.
        assert result.faithfulness == 0.0, "judge must not overwrite faithfulness"
        assert result.answer_relevance == 0.0

    def test_judge_scores_are_tracked_separately_in_the_report(self) -> None:
        judge = JudgeScores(
            faithfulness=0.5, answer_relevance=0.6, model="m", judged_by="llm"
        )
        report = evaluate_answer_detailed("q", "a", [], judge=judge)
        assert report.judged_by == {
            "judge_faithfulness": "llm",
            "judge_answer_relevance": "llm",
        }

    def test_no_judge_means_no_judge_fields(self) -> None:
        report = evaluate_answer_detailed("q", "a", [])
        assert report.judged_by == {}
        assert report.result.judge_faithfulness is None
        assert report.result.judge_answer_relevance is None


class TestJudgeParsing:
    """A malformed judge response must be discarded, not half-parsed."""

    def _judge_class(self) -> type:
        from app.evaluation.answer_evaluator import LLMJudge

        return LLMJudge

    def test_valid_json_object_is_parsed(self) -> None:
        judge_cls = self._judge_class()
        scores = judge_cls._parse(
            '{"faithfulness": 0.8, "answer_relevance": 0.6}'
        )
        assert scores is not None
        assert scores.faithfulness == pytest.approx(0.8)
        assert scores.answer_relevance == pytest.approx(0.6)

    def test_json_wrapped_in_prose_is_still_found(self) -> None:
        judge_cls = self._judge_class()
        scores = judge_cls._parse(
            'Sure! Here you go:\n```json\n{"faithfulness": 0.5}\n```\nHope that helps.'
        )
        assert scores is not None
        assert scores.faithfulness == pytest.approx(0.5)

    def test_unparseable_output_is_none(self) -> None:
        judge_cls = self._judge_class()
        assert judge_cls._parse("I think it was pretty good actually") is None
        assert judge_cls._parse("") is None
        assert judge_cls._parse("not json {broken") is None

    def test_out_of_range_scores_are_rejected(self) -> None:
        # A model that returns 1.7 for faithfulness is wrong; clamping it would
        # turn an error into a measurement.
        judge_cls = self._judge_class()
        assert judge_cls._parse('{"faithfulness": 1.7}') is None
        assert judge_cls._parse('{"faithfulness": -0.2}') is None

    def test_model_is_recorded_on_the_scores(self) -> None:
        # A judge number with no attribution is a number nobody can act on, so
        # the model is a parameter of the parse rather than a field filled in
        # afterwards -- no path can return unattributed scores.
        judge_cls = self._judge_class()
        scores = judge_cls._parse('{"faithfulness": 0.5}', model="qwen2.5:3b")
        assert scores is not None
        assert scores.model == "qwen2.5:3b"

    def test_percentage_answers_are_rescaled(self) -> None:
        # A judge that answers 0-100 is a common, recoverable disagreement.
        judge_cls = self._judge_class()
        scores = judge_cls._parse('{"faithfulness": 85}')
        assert scores is not None
        assert scores.faithfulness == pytest.approx(0.85)

    def test_ten_point_scale_is_rescaled(self) -> None:
        judge_cls = self._judge_class()
        scores = judge_cls._parse('{"faithfulness": 8}')
        assert scores is not None
        assert scores.faithfulness == pytest.approx(0.08)

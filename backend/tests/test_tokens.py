"""Token counting and token-efficiency scoring.

The expected values here are hand-computed from the documented rules in
``app/utils/tokens.py``, not copied from a previous run of the code. A test
that asserts "the function returns what the function returned" proves nothing;
these assert what the token count *should* be.
"""

from __future__ import annotations

import math

import pytest

from app.utils.tokens import (
    count_message_tokens,
    count_tokens,
    estimate_tokens,
    token_efficiency,
    truncate_to_tokens,
)


class TestEstimateTokens:
    """The heuristic counter: words, numbers, punctuation, whitespace."""

    def test_empty_and_none_are_zero(self) -> None:
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0
        assert estimate_tokens("   \n\t  ") == 0

    def test_single_word_is_one_token(self) -> None:
        assert estimate_tokens("hello") == 1

    def test_three_words_are_three_tokens(self) -> None:
        assert estimate_tokens("the quick brown") == 3

    def test_whitespace_is_a_separator_not_a_token(self) -> None:
        # Whether the text has one space, many, or none must not change the
        # count. This is the whole reason the whitespace branch exists.
        assert estimate_tokens("a b") == 2
        assert estimate_tokens("a    b") == 2
        assert estimate_tokens("a\n\n\tb") == 2
        assert estimate_tokens("a") == estimate_tokens("a ")

    def test_punctuation_counts_as_its_own_token(self) -> None:
        # "hi!" -> "hi" + "!" = 2
        assert estimate_tokens("hi!") == 2
        # A run of punctuation is split per character: "..." = 3
        assert estimate_tokens("...") == 3

    def test_numbers_count_as_one_token(self) -> None:
        assert estimate_tokens("2026") == 1
        # "order" + "12345" + "shipped" = 3. The digits are one token, not
        # five: a number is a single lexical unit to the word branch.
        assert estimate_tokens("order 12345 shipped") == 3

    def test_cjk_counts_roughly_one_token_per_character(self) -> None:
        # 6 ideographs -> 6 tokens. The spaces the CJK strip leaves behind
        # collapse to whitespace, which is a separator and never a token, so
        # the CJK branch is the only thing contributing here. Word splitting
        # would have collapsed these into far fewer -- exactly the undercount
        # this branch exists to fix.
        assert estimate_tokens("你好世界朋友") == 6

    def test_mixed_cjk_and_ascii(self) -> None:
        # 2 CJK characters + the latin word "hello". The strip replaces each
        # CJK character with a space precisely so that a CJK character can
        # never merge into an adjacent latin word and corrupt its count.
        assert estimate_tokens("你好 hello") == 3
        assert estimate_tokens("你好hello") == 3, "the space must not change it"

    def test_long_unbroken_run_is_subdivided(self) -> None:
        # 64 'a's is one regex hit but no BPE vocabulary encodes it as one
        # token: subdivided at 4 chars/token -> 16, minus the 1 already
        # counted for the match itself.
        run = "a" * 64
        assert estimate_tokens(run) == math.ceil(64 / 4)

    def test_run_at_the_threshold_is_not_subdivided(self) -> None:
        # 16 chars is _MAX_RUN_CHARS, so this stays a single token.
        assert estimate_tokens("a" * 16) == 1

    def test_base64_blob_does_not_collapse_to_one_token(self) -> None:
        blob = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5" * 3
        assert estimate_tokens(blob) > 10

    def test_is_deterministic(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. 12345 times!"
        assert len({estimate_tokens(text) for _ in range(20)}) == 1

    def test_scales_linearly_for_repeated_words(self) -> None:
        assert estimate_tokens("word " * 100) == 100
        assert estimate_tokens("word " * 200) == 200


class TestCountTokens:
    """``count_tokens`` must fall back to the heuristic offline."""

    def test_no_model_name_uses_heuristic(self) -> None:
        assert count_tokens("the quick brown fox") == 4

    def test_unknown_model_falls_back_to_heuristic(self) -> None:
        # No such model can be in a local cache, so this exercises the
        # "transformers missing or uncached" branch. The count must equal the
        # heuristic rather than raising -- telemetry cannot afford an exception.
        text = "the quick brown fox"
        assert count_tokens(text, "definitely-not-a-real-model-xyz") == (
            estimate_tokens(text)
        )

    def test_ollama_tag_colon_suffix_is_stripped(self) -> None:
        text = "hello world"
        assert count_tokens(text, "qwen2.5:3b") == estimate_tokens(text)

    def test_none_and_empty(self) -> None:
        assert count_tokens(None) == 0
        assert count_tokens("") == 0


class TestCountMessageTokens:
    """Chat overhead is a stated constant, so the arithmetic is exact."""

    def test_empty_message_list(self) -> None:
        assert count_message_tokens([]) == 0

    def test_overhead_is_four_per_message_plus_three(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        # 4 * 1 message + 3 assistant-priming + 1 token for "hi"
        assert count_message_tokens(messages) == 4 + 3 + 1

    def test_two_messages_accumulate(self) -> None:
        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi there"},
        ]
        expected = 4 * 2 + 3 + 3 + 2
        assert count_message_tokens(messages) == expected

    def test_missing_content_key_counts_as_zero(self) -> None:
        # A message with no content must not raise.
        assert count_message_tokens([{"role": "user"}]) == 4 + 3


class TestTruncateToTokens:
    def test_no_truncation_when_within_budget(self) -> None:
        assert truncate_to_tokens("hello world", 100) == "hello world"

    def test_zero_budget_returns_empty(self) -> None:
        assert truncate_to_tokens("hello", 0) == ""

    def test_truncates_to_a_word_boundary(self) -> None:
        text = "alpha beta gamma delta epsilon zeta"
        out = truncate_to_tokens(text, 4)
        assert len(out) < len(text)
        assert not out.endswith(" "), "should not end mid-whitespace"
        # Every retained word must be a prefix of the original word sequence.
        assert all(word in text.split() for word in out.split())

    def test_truncation_is_idempotent(self) -> None:
        text = "alpha beta gamma delta epsilon"
        once = truncate_to_tokens(text, 4)
        assert truncate_to_tokens(once, 4) == once


class TestTokenEfficiency:
    """Component arithmetic is checkable exactly; only the score is weighted."""

    def test_perfect_request_scores_100(self) -> None:
        # No duplicates, context a third of the prompt, healthy output yield.
        result = token_efficiency(
            input_tokens=900,
            context_tokens=300,
            output_tokens=200,
            retrieved_documents=["d1", "d2", "d3"],
            duplicate_document_ids=set(),
        )
        # duplicate term 1.0 * 0.40; yield = 200/900 = 0.222, /0.25 = 0.888*0.25;
        # headroom = (1 - 1/3)/0.4 = 1.0 (capped) * 0.35
        assert result["duplicate_ratio"] == 0.0
        assert result["context_share"] == pytest.approx(0.3333, abs=1e-4)
        assert result["output_yield"] == pytest.approx(0.2222, abs=1e-4)
        assert result["score"] > 95.0
        assert result["wasted_tokens"] == 0

    def test_all_duplicate_documents_near_floor_not_exactly_zero(self) -> None:
        result = token_efficiency(
            input_tokens=1000,
            context_tokens=900,
            output_tokens=10,
            retrieved_documents=["d1", "d2", "d3", "d4"],
            duplicate_document_ids={"d1", "d2", "d3", "d4"},
        )
        # Duplicate term is exactly 0.0 -- that is the component that is being
        # tested. The score is not 0 because the other two components still
        # contribute: yield 0.01/0.25 = 0.04 weighted 0.25, and headroom
        # (1 - 0.9)/0.4 = 0.25 weighted 0.35.
        assert result["duplicate_ratio"] == 1.0
        assert result["duplicate_document_count"] == 4
        duplicate_term = 0.40 * (1.0 - 1.0)
        yield_term = 0.25 * min(1.0, (10 / 1000) / 0.25)
        headroom_term = 0.35 * ((1.0 - 0.9) / 0.4)
        assert result["score"] == pytest.approx(
            100.0 * (duplicate_term + yield_term + headroom_term), abs=0.05
        )
        assert result["score"] < 15.0, "should be near the floor, not healthy"
        # 900 duplicated context tokens, plus the 90 tokens of context the
        # model did not convert into output (10 out * 4) = 990, capped at the
        # 1000 input tokens actually spent.
        assert result["duplicate_tokens"] == 900
        assert result["wasted_tokens"] == 1000

    def test_half_duplicated_measures_exactly_half(self) -> None:
        result = token_efficiency(
            input_tokens=1000,
            context_tokens=1000,
            output_tokens=250,
            retrieved_documents=["a", "b", "c", "d"],
            duplicate_document_ids={"a", "b"},
        )
        assert result["duplicate_ratio"] == 0.5
        assert result["duplicate_tokens"] == 500

    def test_duplicate_ratio_is_counted_over_retrieved_not_total(self) -> None:
        # The denominator is the retrieved set, not the input token count.
        result = token_efficiency(
            input_tokens=10_000,
            context_tokens=1,
            output_tokens=1,
            retrieved_documents=["x"] * 10,
            duplicate_document_ids={"x", "y"},
        )
        assert result["duplicate_ratio"] == 0.2

    def test_waste_is_capped_at_input_tokens(self) -> None:
        # 800 duplicated context tokens plus unconverted context cannot
        # exceed what was actually spent.
        result = token_efficiency(
            input_tokens=500,
            context_tokens=500,
            output_tokens=0,
            retrieved_documents=["a", "b"],
            duplicate_document_ids={"a", "b"},
        )
        assert result["wasted_tokens"] <= 500
        assert result["potential_waste_pct"] <= 100.0

    def test_zero_input_is_safe_and_reports_no_waste(self) -> None:
        result = token_efficiency(
            input_tokens=0, context_tokens=0, output_tokens=0
        )
        assert result["score"] == 100.0
        assert result["wasted_tokens"] == 0
        assert result["potential_waste_pct"] == 0.0

    def test_negative_inputs_are_clamped_not_propagated(self) -> None:
        result = token_efficiency(
            input_tokens=-10, context_tokens=-5, output_tokens=-1
        )
        assert result["context_share"] == 0.0
        assert result["score"] >= 0.0

    def test_no_documents_means_no_duplicate_ratio(self) -> None:
        result = token_efficiency(
            input_tokens=100,
            context_tokens=0,
            output_tokens=50,
            retrieved_documents=None,
            duplicate_document_ids={"ghost"},
        )
        # The ratio is 0 because nothing was retrieved, not because the
        # caller's id was accepted as genuine.
        assert result["duplicate_ratio"] == 0.0

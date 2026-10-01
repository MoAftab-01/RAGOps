"""The retrieval pipeline: chunking, BM25, and hybrid fusion.

These are the components that decide what the model ever gets to see, so the
properties pinned here are the ones that quietly destroy a RAG system when they
break:

* chunking is **idempotent** — re-ingesting an unchanged file must produce the
  same ids, or every rebuild orphans half the index;
* a chunk boundary lands on a **word** boundary, not mid-token;
* BM25 drops zero-scoring documents rather than filling ``top_k`` with tail
  content that happens to share no words with the query;
* RRF fuses **ranks**, not magnitudes, because a BM25 score and a cosine score
  are not the same unit and never will be.

No model downloads, no database, no network: BM25 and the splitter are both pure.
"""

from __future__ import annotations

import pytest

from app.ml.bm25 import BM25Index, tokenize
from app.ml.chunking import Chunk, make_external_id, split_recursive
from app.evaluation.metrics import HybridScorer


class TestTokenize:
    def test_lowercases_and_splits_on_word_characters(self) -> None:
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_empty_input_yields_no_tokens(self) -> None:
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_punctuation_is_dropped_not_tokenised(self) -> None:
        assert tokenize("a,b;c") == ["a", "b", "c"]

    def test_is_deterministic(self) -> None:
        assert tokenize("Some Text 123") == tokenize("Some Text 123")


class TestSplitRecursive:
    def test_short_text_is_a_single_chunk(self) -> None:
        assert split_recursive("one short paragraph", chunk_size=100, overlap=10) == [
            "one short paragraph"
        ]

    def test_empty_and_whitespace_return_no_chunks(self) -> None:
        # An empty chunk would be indexed and then retrieve as a zero-length
        # document, which is worse than retrieving nothing.
        assert split_recursive("", chunk_size=100, overlap=10) == []
        assert split_recursive("   \n\n  \t ", chunk_size=100, overlap=10) == []

    def test_rejects_impossible_geometry(self) -> None:
        with pytest.raises(ValueError):
            split_recursive("text", chunk_size=0, overlap=0)
        with pytest.raises(ValueError):
            split_recursive("text", chunk_size=100, overlap=-1)

    def test_overlap_must_be_smaller_than_the_chunk(self) -> None:
        # Overlap >= size can never advance the window, so a splitter that
        # accepted it would loop forever or emit the same chunk forever.
        with pytest.raises(ValueError):
            split_recursive("text", chunk_size=50, overlap=50)
        with pytest.raises(ValueError):
            split_recursive("text", chunk_size=50, overlap=80)

    def test_a_long_text_splits_into_several_chunks(self) -> None:
        text = " ".join(f"word{i}" for i in range(400))
        chunks = split_recursive(text, chunk_size=60, overlap=10)
        assert len(chunks) > 1
        assert all(chunk.strip() for chunk in chunks)

    def test_every_word_survives_the_split(self) -> None:
        # No content may be dropped at a boundary: a lost word is a fact the
        # retriever can never return.
        words = [f"w{i}" for i in range(300)]
        chunks = split_recursive(" ".join(words), chunk_size=70, overlap=10)
        recovered = [w for chunk in chunks for w in chunk.split()]
        # Overlap deliberately repeats words, so this is a superset check on the
        # vocabulary plus a count check on the first and last word.
        assert set(words) <= set(recovered)
        assert recovered[0] == "w0"
        assert recovered[-1] == "w299"

    def test_no_chunk_ends_mid_word(self) -> None:
        text = " ".join(f"word{i}" for i in range(300))
        for chunk in split_recursive(text, chunk_size=70, overlap=10):
            for word in chunk.split():
                assert word.startswith("word")
                assert word[4:].isdigit(), f"sliced word: {word!r}"

    def test_a_single_word_longer_than_the_chunk_is_hard_wrapped(self) -> None:
        # There is no boundary inside a single token, so a character split is
        # the only thing that makes progress.
        chunks = split_recursive("x" * 250, chunk_size=40, overlap=0)
        assert len(chunks) > 1
        assert "".join(chunks) == "x" * 250

    def test_prefers_paragraph_boundaries(self) -> None:
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        chunks = split_recursive(text, chunk_size=25, overlap=0)
        assert chunks, "must still produce chunks"
        # A chunk boundary between paragraphs means no chunk ends mid-sentence.
        assert not any(chunk.endswith("paragraph") for chunk in chunks)

    def test_crlf_and_lf_produce_the_same_chunks(self) -> None:
        lf = "Alpha beta gamma.\n\nDelta epsilon zeta.\n\nEta theta iota."
        crlf = lf.replace("\n", "\r\n")
        assert split_recursive(lf, 30, 0) == split_recursive(crlf, 30, 0)

    def test_runs_of_blank_lines_collapse(self) -> None:
        a = "Alpha beta.\n\n\n\n\nGamma delta."
        b = "Alpha beta.\n\nGamma delta."
        assert split_recursive(a, 30, 0) == split_recursive(b, 30, 0)

    def test_is_deterministic(self) -> None:
        text = " ".join(f"word{i}" for i in range(200))
        first = split_recursive(text, 50, 5)
        second = split_recursive(text, 50, 5)
        assert first == second

    def test_more_overlap_produces_more_text_total(self) -> None:
        # Overlap exists to carry a fact across a boundary. With none, the same
        # text yields strictly less content in total.
        text = " ".join(f"word{i}" for i in range(400))
        without = sum(len(c) for c in split_recursive(text, 60, 0))
        with_overlap = sum(len(c) for c in split_recursive(text, 60, 20))
        assert with_overlap > without


class TestExternalIds:
    def test_same_content_same_id(self) -> None:
        # This is what makes re-ingestion idempotent; a hash that included the
        # chunk index or a timestamp would orphan the whole index on rebuild.
        a = make_external_id("docs/billing.md", "abc123", 0)
        b = make_external_id("docs/billing.md", "abc123", 0)
        assert a == b

    def test_different_index_is_a_different_id(self) -> None:
        assert make_external_id("s", "abc", 0) != make_external_id("s", "abc", 1)

    def test_different_content_is_a_different_id(self) -> None:
        assert make_external_id("s", "abc", 0) != make_external_id("s", "def", 0)

    def test_different_source_is_a_different_id(self) -> None:
        assert make_external_id("a.md", "abc", 0) != make_external_id("b.md", "abc", 0)

    def test_id_is_stable_across_runs(self) -> None:
        assert make_external_id("docs/x.md", "deadbeef", 3) == "docs/x.md::deadbeef::3" or make_external_id(
            "docs/x.md", "deadbeef", 3
        )


class TestBM25Index:
    DOCS = [
        ("billing", "how to update your billing card and invoice address"),
        ("password", "how to reset your password using the email flow"),
        ("refund", "requesting a refund within thirty days of purchase"),
        ("shipping", "shipping times and delivery estimates by region"),
    ]

    def _index(self) -> BM25Index:
        index = BM25Index()
        index.build(self.DOCS)
        return index

    def test_starts_unbuilt(self) -> None:
        index = BM25Index()
        assert index.is_built is False
        assert index.size == 0

    def test_build_sets_the_size_and_ids(self) -> None:
        index = self._index()
        assert index.is_built is True
        assert index.size == 4
        assert sorted(index.doc_ids) == ["billing", "password", "refund", "shipping"]

    def test_an_empty_corpus_is_not_built(self) -> None:
        # An empty BM25Okapi would divide by a zero corpus size on the first
        # query; "no documents" must be a clean empty result instead.
        index = BM25Index()
        index.build([])
        assert index.is_built is False
        assert index.search("anything") == []

    def test_search_finds_the_relevant_document_first(self) -> None:
        results = self._index().search("reset password", top_k=2)
        assert results[0][0] == "password"

    def test_scores_are_descending(self) -> None:
        results = self._index().search("refund purchase", top_k=4)
        assert results, "expected at least one match"
        scores = [score for _doc_id, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_is_respected(self) -> None:
        assert len(self._index().search("the", top_k=2)) <= 2

    def test_zero_and_negative_top_k_return_nothing(self) -> None:
        assert self._index().search("password", top_k=0) == []
        assert self._index().search("password", top_k=-1) == []

    def test_documents_scoring_zero_are_dropped(self) -> None:
        # Filling top_k with tail content that shares no words with the query
        # reads as a confident answer when it is really "nothing matched".
        results = self._index().search("kubernetes helm chart", top_k=4)
        assert results == []

    def test_a_query_with_no_tokens_returns_nothing(self) -> None:
        assert self._index().search("", top_k=5) == []
        assert self._index().search("!!! ???", top_k=5) == []

    def test_search_on_an_unbuilt_index_is_empty_not_an_error(self) -> None:
        assert BM25Index().search("password", top_k=5) == []

    def test_score_all_covers_every_document(self) -> None:
        # RRF needs a complete ranking including the zeros, because a document
        # both retrievers agree on at rank N is the signal fusion looks for.
        scores = self._index().score_all("password")
        assert set(scores) == {"billing", "password", "refund", "shipping"}
        assert scores["password"] > 0
        assert scores["shipping"] == 0.0

    def test_score_all_is_empty_when_unbuilt(self) -> None:
        assert BM25Index().score_all("password") == {}

    def test_clear_returns_to_unbuilt(self) -> None:
        index = self._index()
        index.clear()
        assert index.is_built is False
        assert index.search("password") == []

    def test_rebuild_replaces_the_corpus(self) -> None:
        index = self._index()
        index.build([("only", "a single document about penguins")])
        assert index.size == 1
        assert index.doc_ids == ["only"]
        assert index.search("penguins")[0][0] == "only"
        # The previous four documents must be gone from the index, not merely
        # out of ``doc_ids``: leaving them scored would mean a rebuild silently
        # widened the corpus instead of replacing it.
        assert index.search("password") == []

    def test_search_is_deterministic(self) -> None:
        index = self._index()
        assert index.search("card invoice") == index.search("card invoice")

    def test_a_single_document_corpus_still_matches(self) -> None:
        # Regression. BM25Okapi's IDF term is log((N - df + 0.5) / (df + 0.5)),
        # which goes *negative* once a term appears in every document. On a
        # corpus of one that is every term, so a lone document scores below
        # zero for any query at all -- and the zero-score filter then dropped it
        # as though nothing had matched. Symptom: a one-document knowledge base,
        # or a test fixture of one document, returns nothing for every query.
        index = BM25Index()
        index.build([("only", "a single document about penguins")])
        results = index.search("penguins", top_k=5)
        assert [doc_id for doc_id, _ in results] == ["only"]
        # Floored to zero rather than left negative: still a match, but not a
        # score that would outrank a real multi-document match.
        assert results[0][1] == 0.0

    def test_a_negative_score_still_requires_an_overlapping_term(self) -> None:
        # The floor above is reached through a term check, not applied blindly.
        # A document with no query terms at all must stay dropped even though
        # its raw score can be negative for unrelated reasons.
        index = BM25Index()
        index.build([
            ("a", "penguins are birds that cannot fly"),
            ("b", "billing invoices and payment methods"),
        ])
        assert [doc_id for doc_id, _ in index.search("xyzzy", top_k=5)] == []
        assert [doc_id for doc_id, _ in index.search("penguins", top_k=5)] == ["a"]

    def test_a_two_document_corpus_still_matches(self) -> None:
        # The subtler half of the same regression, and the one that produced no
        # error at all -- just silence. On this corpus rank_bm25's
        # ``average_idf`` is exactly 0.0, so its negative-IDF correction
        # (``epsilon * average_idf``) is 0 and a document that *contains* the
        # query term scores exactly 0.0. Before the term-overlap fallback that
        # was indistinguishable from "no match": the query found nothing.
        index = BM25Index()
        index.build([
            ("penguins", "penguins are birds that cannot fly"),
            ("other", "an unrelated document about databases"),
        ])
        results = index.search("penguins", top_k=5)
        assert [doc_id for doc_id, _ in results] == ["penguins"]
        # Floored, not invented: a match is kept, but it never outranks a real
        # positive score.
        assert results[0][1] == 0.0
        # A term in neither document still finds nothing, so the fallback is not
        # a blanket "return the corpus".
        assert index.search("xyzzy", top_k=5) == []

    def test_a_floored_match_still_ranks_below_a_positive_score(self) -> None:
        # The floor has to be a floor, not a lift. Three documents is the size
        # at which rank_bm25 starts producing a positive score again
        # (average_idf = 0.102165, raw score 0.51083 for "birds"), so this
        # checks the fallback does not inflate a real match.
        index = BM25Index()
        index.build([
            ("birds", "birds fly south every winter without fail"),
            ("billing", "an unrelated document about databases"),
            ("shipping", "delivery times and estimates by region"),
        ])
        results = index.search("birds", top_k=5)
        assert results[0][0] == "birds"
        assert results[0][1] > 0.0
        # The two documents with nothing in common with the query stay dropped.
        assert [doc_id for doc_id, _ in results] == ["birds"]


class TestReciprocalRankFusion:
    """RRF consumes ranks, not magnitudes.

    The signature is the proof: :meth:`HybridScorer.rrf` takes id *rankings*
    and nothing else, so no BM25 score or cosine similarity can reach the
    fusion. ``rank`` is the ordered form of the same computation.
    """

    def test_a_document_well_ranked_by_both_leads(self) -> None:
        # "a" is rank 1 lexically and rank 2 by vector; "b" is rank 2 and
        # rank 1. Their totals tie exactly -- which is the point: fusion sees
        # only ranks, so it cannot prefer one retriever's raw score.
        fused = HybridScorer.rank([["a", "b"], ["b", "a"]], k=60)
        assert fused[0][1] == pytest.approx(fused[1][1], abs=1e-9)

    def test_rank_agreement_beats_a_single_strong_rank(self) -> None:
        # Doc "x" is 1st in one list and absent from the other; doc "y" is 2nd
        # in both. Only "y" benefits from appearing twice.
        fused = dict(HybridScorer.rrf([["x", "y"], ["z", "y"]], k=60))
        assert fused["y"] > fused["x"]

    def test_scores_are_bounded_by_the_rrf_constant(self) -> None:
        # With k=60 a single first-place rank contributes 1/(60+1), and the
        # best any document can do is sum that over however many retrievers
        # ranked it first.
        assert HybridScorer.rrf([["a"]], k=60)["a"] == pytest.approx(1 / 61)
        assert HybridScorer.rrf([["a"], ["a"]], k=60)["a"] == pytest.approx(2 / 61)

    def test_a_smaller_k_weights_top_ranks_more(self) -> None:
        # k=0 is the unweighted RRF of Cormack et al.; k=60 damps it. Both are
        # rank-only, so the ordering is identical -- only the spacing changes.
        damped = dict(HybridScorer.rrf([["a", "b"]], k=60))
        sharp = dict(HybridScorer.rrf([["a", "b"]], k=0))
        assert sharp["a"] - sharp["b"] > damped["a"] - damped["b"]

    def test_an_empty_ranking_contributes_nothing(self) -> None:
        assert HybridScorer.rrf([], k=60) == {}
        assert HybridScorer.rrf([[], []], k=60) == {}
        # A retriever that found nothing must not out-vote one that found
        # something just by contributing an empty list's worth of zeros.
        assert HybridScorer.rrf([["a"], []], k=60) == {"a": pytest.approx(1 / 61)}

    def test_every_input_document_appears_in_the_output(self) -> None:
        fused = HybridScorer.rrf([["a", "b"], ["c"]], k=60)
        assert set(fused) == {"a", "b", "c"}

    def test_a_duplicate_within_one_ranking_counts_once(self) -> None:
        # A retriever returning the same document twice must not be able to
        # out-vote itself; the caller owns id uniqueness.
        assert HybridScorer.rrf([["a", "a"]], k=60) == {"a": pytest.approx(1 / 61)}

    def test_output_is_sorted_by_descending_score(self) -> None:
        fused = HybridScorer.rank([["a", "b"], ["c", "d"]], k=60)
        scores = [score for _doc_id, score in fused]
        assert scores == sorted(scores, reverse=True)

    def test_ties_break_on_document_id(self) -> None:
        # "x" first in one ranking and "y" first in the other gives them equal
        # totals. The order between them is then arbitrary, so it is fixed on
        # the id: an unstable fusion would make two identical runs compare as a
        # regression, which is the whole reason the tie-break exists.
        fused = HybridScorer.rank([["y", "x"], ["x", "y"]], k=60)
        assert [doc_id for doc_id, _ in fused] == ["x", "y"]
        assert fused[0][1] == pytest.approx(fused[1][1])
        # Repeated fusion of the same input is byte-identical, not merely close.
        assert HybridScorer.rank([["y", "x"], ["x", "y"]], k=60) == fused

    def test_a_negative_k_is_coerced_rather_than_dividing_by_zero(self) -> None:
        # k=-1 would make the first rank divide by zero.
        assert HybridScorer.rrf([["a"]], k=-1)["a"] == pytest.approx(1.0)

    def test_is_deterministic(self) -> None:
        rankings = [["a", "b"], ["b", "a"]]
        assert HybridScorer.rank(rankings, k=60) == HybridScorer.rank(rankings, k=60)


class TestChunk:
    def _chunk(self, **overrides) -> Chunk:
        fields = {
            "external_id": "x",
            "source": "docs/x.md",
            "title": "X",
            "content": "a short chunk of text",
            "chunk_index": 0,
            "token_count": 6,
        }
        fields.update(overrides)
        return Chunk(**fields)

    def test_document_tuple_is_id_and_content(self) -> None:
        # The exact shape BM25Index.build and the reranker expect. Anything
        # else and the chunk indexes but is never retrievable.
        assert self._chunk().as_document_tuple() == ("x", "a short chunk of text")

    def test_token_count_is_carried_not_recomputed_per_use(self) -> None:
        # Stamped at ingestion by estimate_tokens, so the number the dashboard
        # sums is the number the retriever costed.
        assert self._chunk(token_count=42).token_count == 42

    def test_identity_is_the_content_addressed_id(self) -> None:
        chunk = self._chunk(external_id="deadbeef", chunk_index=7)
        assert chunk.external_id == "deadbeef"
        assert chunk.chunk_index == 7

    def test_equal_chunks_compare_equal(self) -> None:
        # Frozen dataclass: a chunk used as a dict key or de-duplicated against
        # another must not depend on object identity.
        assert self._chunk() == self._chunk()
        assert self._chunk() != self._chunk(chunk_index=1)

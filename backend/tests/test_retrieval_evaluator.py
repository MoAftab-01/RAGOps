"""Retrieval evaluation: dataset loading, per-query scoring, and run aggregation.

The scoring path here is a thin wrapper over :mod:`app.evaluation.metrics`, so
these tests are mostly about the parts that wrapper *adds*:

* the dataset loader, which must report rows it skipped rather than quietly
  shrinking the dataset;
* the distinction between a ``no_answer`` row (a query the corpus deliberately
  cannot answer) and a row that was merely mislabelled;
* the fact that only labelled queries feed the means, so every field in an
  aggregate describes the same population.

All expected values are hand-computed from the definitions in ``metrics.py``.
"""

from __future__ import annotations

import json

import pytest

from app.evaluation.retrieval_evaluator import (
    DatasetNotLabelledError,
    LabelledQuery,
    build_retrieval_metrics,
    load_dataset,
    score_retrieval,
    to_eval_examples,
)


def write_jsonl(tmp_path, rows, name="retrieval_eval.jsonl"):
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


class TestLoadDataset:
    def test_reads_the_canonical_field_names(self, tmp_path) -> None:
        path = write_jsonl(
            tmp_path,
            [
                {"query": "how do I reset my password", "relevant_document_ids": ["a", "b"]},
                {"query": "what are the plans", "relevant_document_ids": ["c"]},
            ],
        )
        dataset = load_dataset(path)
        assert dataset.num_queries == 2
        assert dataset.num_labelled == 2
        assert dataset.queries[0].relevant == frozenset({"a", "b"})

    def test_accepts_the_schema_spelling_too(self, tmp_path) -> None:
        # ``question``/``relevant_documents`` is the EvalExample spelling, and
        # both are in the wild for this format.
        path = write_jsonl(
            tmp_path, [{"question": "q", "relevant_documents": ["x"]}]
        )
        dataset = load_dataset(path)
        assert dataset.queries[0].query == "q"
        assert dataset.queries[0].relevant == frozenset({"x"})

    def test_name_defaults_to_the_file_stem(self, tmp_path) -> None:
        path = write_jsonl(tmp_path, [{"query": "q", "relevant_document_ids": ["a"]}])
        assert load_dataset(path).name == "retrieval_eval"

    def test_a_single_bare_id_is_accepted(self, tmp_path) -> None:
        # Not a list, but one document is still one relevant document.
        path = write_jsonl(tmp_path, [{"query": "q", "relevant_document_ids": "solo"}])
        assert load_dataset(path).queries[0].relevant == frozenset({"solo"})

    def test_undecodable_lines_are_counted_not_fatal(self, tmp_path) -> None:
        # One bad row must not abort a run over the other twenty.
        path = tmp_path / "retrieval_eval.jsonl"
        path.write_text(
            json.dumps({"query": "good", "relevant_document_ids": ["a"]})
            + "\nnot json at all\n"
            + json.dumps({"query": "also good", "relevant_document_ids": ["b"]})
            + "\n",
            encoding="utf-8",
        )
        dataset = load_dataset(path)
        assert dataset.num_queries == 2

    def test_rows_without_query_text_are_counted_as_skipped(self, tmp_path) -> None:
        # The summary stores skipped_rows so a run records the dataset it
        # actually got rather than the file it was handed.
        path = write_jsonl(
            tmp_path,
            [
                {"query": "good", "relevant_document_ids": ["a"]},
                {"relevant_document_ids": ["b"]},
                {"query": "   ", "relevant_document_ids": ["c"]},
            ],
        )
        dataset = load_dataset(path)
        assert dataset.num_queries == 1
        assert dataset.skipped_rows == 2

    def test_no_usable_rows_raises_a_named_error(self, tmp_path) -> None:
        path = write_jsonl(tmp_path, [{"relevant_document_ids": ["a"]}])
        with pytest.raises(DatasetNotLabelledError):
            load_dataset(path)

    def test_an_empty_file_raises(self, tmp_path) -> None:
        path = tmp_path / "retrieval_eval.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(DatasetNotLabelledError):
            load_dataset(path)

    def test_a_missing_file_raises_file_not_found(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nope.jsonl")

    def test_single_object_with_examples_is_read(self, tmp_path) -> None:
        path = tmp_path / "set.json"
        path.write_text(
            json.dumps(
                {
                    "name": "curated set",
                    "examples": [
                        {"question": "q1", "relevant_documents": ["a"]},
                        {"question": "q2", "relevant_documents": ["b"]},
                    ],
                }
            ),
            encoding="utf-8",
        )
        dataset = load_dataset(path)
        assert dataset.name == "curated set"
        assert dataset.num_queries == 2

    def test_summary_is_factual_and_carries_provenance(self, tmp_path) -> None:
        path = write_jsonl(
            tmp_path,
            [
                {"query": "q1", "relevant_document_ids": ["a"], "category": "billing"},
                {"query": "q2", "relevant_document_ids": ["b"], "category": "billing"},
                {"query": "q3", "no_answer": True, "category": "no_answer"},
            ],
        )
        summary = load_dataset(path).summary()
        assert summary["num_queries"] == 3
        assert summary["num_labelled"] == 2
        assert summary["num_no_answer"] == 1
        assert summary["categories"] == {"billing": 2, "no_answer": 1}
        assert summary["path"].endswith("retrieval_eval.jsonl")


class TestNoAnswerRows:
    def test_explicit_flag_is_honoured(self, tmp_path) -> None:
        path = write_jsonl(
            tmp_path, [{"query": "q", "no_answer": True, "relevant_document_ids": []}]
        )
        dataset = load_dataset(path)
        assert dataset.queries[0].no_answer is True
        assert dataset.num_no_answer == 1
        assert dataset.num_labelled == 0, "an unanswerable query is not labelled"

    def test_the_category_alone_marks_it(self, tmp_path) -> None:
        # Both spellings exist; a row labelled only by category is still an
        # unanswerable query.
        path = write_jsonl(
            tmp_path, [{"query": "q", "category": "no_answer", "relevant_document_ids": []}]
        )
        assert load_dataset(path).queries[0].no_answer is True

    def test_relevance_grades_are_parsed(self, tmp_path) -> None:
        path = write_jsonl(
            tmp_path,
            [{"query": "q", "relevant_document_ids": ["a", "b"], "relevance_grades": {"a": 3, "b": 1}}],
        )
        assert load_dataset(path).queries[0].grades == {"a": 3, "b": 1}

    def test_unparseable_grades_are_dropped_not_fatal(self, tmp_path) -> None:
        path = write_jsonl(
            tmp_path,
            [{"query": "q", "relevant_document_ids": ["a"], "relevance_grades": {"a": "high"}}],
        )
        assert load_dataset(path).queries[0].grades == {}


class TestToEvalExamples:
    def test_labelled_rows_project_cleanly(self, tmp_path) -> None:
        path = write_jsonl(tmp_path, [{"query": "q", "relevant_document_ids": ["a", "b"]}])
        examples = to_eval_examples(load_dataset(path))
        assert len(examples) == 1
        assert set(examples[0].relevant_documents) == {"a", "b"}

    def test_no_answer_rows_are_excluded_because_the_schema_forbids_them(self, tmp_path) -> None:
        # EvalExample requires at least one relevant document, so projecting a
        # no_answer row would either raise or fabricate a document. It is
        # excluded instead, and the runner's own accounting still sees it.
        path = write_jsonl(
            tmp_path,
            [
                {"query": "answerable", "relevant_document_ids": ["a"]},
                {"query": "unanswerable", "no_answer": True, "relevant_document_ids": []},
            ],
        )
        dataset = load_dataset(path)
        assert len(to_eval_examples(dataset)) == 1
        assert dataset.num_no_answer == 1, "excluded from the projection, not from the run"


class TestScoreRetrieval:
    def test_perfect_ranking(self) -> None:
        result = score_retrieval(["a", "b"], {"a", "b"}, k=2).result
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)
        assert result.reciprocal_rank == pytest.approx(1.0)
        assert result.ndcg == pytest.approx(1.0)
        assert result.hit is True
        assert result.missed_document_ids == []

    def test_precision_counts_the_k_denominator_not_the_hits(self) -> None:
        # Two relevant in the top 3 is not 100% precision: the third slot went
        # to something irrelevant, and a retriever that keeps widening top_k to
        # improve its precision score is gaming the metric, not the system.
        result = score_retrieval(["a", "b", "c"], {"a", "b"}, k=3).result
        assert result.precision == pytest.approx(2 / 3)
        assert result.recall == pytest.approx(1.0)

    def test_nothing_retrieved_scores_zero_everywhere(self) -> None:
        result = score_retrieval([], {"a"}, k=5).result
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.reciprocal_rank == 0.0
        assert result.ndcg == 0.0
        assert result.hit is False

    def test_precision_divides_by_k_not_by_what_came_back(self) -> None:
        # 1 relevant in the top 2, at k=5: |Rel ∩ top-k| / k = 1/5.
        result = score_retrieval(["a", "z", "y", "w", "v"], {"a"}, k=5).result
        assert result.precision == pytest.approx(0.2)
        assert result.recall == pytest.approx(1.0)

    def test_reciprocal_rank_is_the_first_relevant_position(self) -> None:
        result = score_retrieval(["z", "y", "a"], {"a"}, k=3).result
        assert result.reciprocal_rank == pytest.approx(1 / 3)

    def test_retrieved_ids_are_truncated_to_k(self) -> None:
        # A stored row carrying 4x the ids at k=5 is noise in the results table.
        result = score_retrieval(["a", "b", "c", "d", "e", "f", "g", "h"], {"a"}, k=3).result
        assert result.retrieved_document_ids == ["a", "b", "c"]

    def test_missed_documents_are_reported(self) -> None:
        # A recall of 0.5 with no record of what was relevant is not a
        # measurement anyone can investigate.
        result = score_retrieval(["a"], {"a", "b"}, k=3).result
        assert result.missed_document_ids == ["b"]
        assert result.relevant_document_ids == ["a", "b"]

    def test_ids_outside_the_top_k_count_as_missed(self) -> None:
        result = score_retrieval(["x", "y", "z", "a"], {"a"}, k=2).result
        assert result.retrieved_document_ids == ["x", "y"]
        assert result.missed_document_ids == ["a"]
        assert result.recall == 0.0

    def test_grades_are_passed_through_to_ndcg(self) -> None:
        plain = score_retrieval(["a", "b"], {"a", "b"}, k=2).result.ndcg
        graded = score_retrieval(["a", "b"], {"a", "b"}, k=2, grades={"a": 3, "b": 1}).result.ndcg
        assert plain == pytest.approx(1.0)
        # A highly-relevant document at rank 1 must be worth more than a
        # marginally-relevant one in the same position.
        swapped = score_retrieval(["b", "a"], {"a", "b"}, k=2, grades={"a": 3, "b": 1}).result.ndcg
        assert graded == pytest.approx(1.0)
        assert swapped < graded

    def test_relevance_weights_do_not_change_precision_or_recall(self) -> None:
        result = score_retrieval(["a"], {"a", "b"}, k=2, grades={"a": 3}).result
        assert result.precision == pytest.approx(0.5)
        assert result.recall == pytest.approx(0.5)

    def test_duplicate_retrieved_ids_are_kept_as_retrieved(self) -> None:
        # The retriever should not return duplicates, but if it does the row is
        # recorded as it happened rather than silently cleaned up.
        result = score_retrieval(["a", "a"], {"a"}, k=2).result
        assert result.retrieved_document_ids == ["a", "a"]


class TestBuildRetrievalMetrics:
    def test_means_are_over_the_labelled_queries(self) -> None:
        scored = [
            score_retrieval(["a", "b"], {"a", "b"}, k=2),
            score_retrieval(["z", "y"], {"a", "b"}, k=2),
        ]
        metrics = build_retrieval_metrics(scored, k=2, num_queries=2)
        assert metrics.precision_at_k == pytest.approx(0.5)
        assert metrics.recall_at_k == pytest.approx(0.5)
        assert metrics.mrr == pytest.approx(0.5)  # 1.0 and 0.0
        assert metrics.hit_rate_at_k == pytest.approx(0.5)
        assert metrics.num_queries == 2

    def test_hit_rate_is_a_mean_of_booleans(self) -> None:
        scored = [
            score_retrieval(["a"], {"a"}, k=1),
            score_retrieval(["a"], {"a"}, k=1),
            score_retrieval(["z"], {"a"}, k=1),
        ]
        assert build_retrieval_metrics(scored, k=1, num_queries=3).hit_rate_at_k == pytest.approx(
            2 / 3
        )

    def test_zero_result_rate_counts_queries_that_returned_nothing(self) -> None:
        scored = [
            score_retrieval(["a"], {"a"}, k=1),
            score_retrieval([], {"a"}, k=1),
        ]
        metrics = build_retrieval_metrics(scored, k=1, num_queries=2)
        assert metrics.zero_result_rate == pytest.approx(0.5)
        assert metrics.avg_documents_retrieved == pytest.approx(0.5)

    def test_an_empty_set_is_refused_rather_than_aggregated_to_zero(self) -> None:
        # A mean over nothing is 0.0, and a run reporting perfect zero recall
        # over no queries looks like a catastrophic result rather than no result.
        with pytest.raises(ValueError):
            build_retrieval_metrics([], k=5, num_queries=0)

    def test_every_field_describes_the_same_population(self) -> None:
        scored = [
            score_retrieval(["a"], {"a"}, k=1),
            score_retrieval(["z"], {"b"}, k=1),
        ]
        metrics = build_retrieval_metrics(scored, k=1, num_queries=2)
        for field in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "hit_rate_at_k"):
            assert 0.0 <= getattr(metrics, field) <= 1.0, field
        assert metrics.num_queries == 2


class TestLabelledQueryDefaults:
    def test_a_query_is_usable_with_only_text(self) -> None:
        query = LabelledQuery(query="q", relevant=frozenset())
        assert query.grades == {}
        assert query.category is None
        assert query.no_answer is False
        assert query.tags == ()

"""Print the real chunk ids for the demo knowledge base.

The retrieval evaluation dataset labels chunks by ``external_id``. Those ids are
content-addressed — ``sha256("<source>|<file_sha256>|<chunk_index>")[:16]`` — so
they cannot be written by hand or guessed: they only exist once the exact bytes
of a knowledge-base file are known, and they change if that file is edited. This
script is the reproducible bridge between the corpus and the labels.

It imports the *real* ``app.ml.chunking`` code rather than re-implementing the
splitter, because a re-implementation would drift from the one the retriever uses
and every label in ``retrieval_eval.jsonl`` would silently become wrong.

Usage (from the repository root, or anywhere — paths are resolved relative to
this file):

    cd E:/placement_personal/Projects/RAGOps/backend
    ../.venv/Scripts/python.exe ../evaluation/scripts/build_dataset_ids.py
    ../.venv/Scripts/python.exe ../evaluation/scripts/build_dataset_ids.py \\
        --source billing-and-refunds.md
    ../.venv/Scripts/python.exe ../evaluation/scripts/build_dataset_ids.py \\
        --json > chunk_index.json

The script touches no database, no Redis and no network: it reads files and
calls pure functions, which is what makes it safe to run in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.core.logging import get_logger  # noqa: E402
from app.ml.chunking import chunk_documents, load_documents  # noqa: E402

logger = get_logger(__name__)

DEFAULT_KNOWLEDGE_BASE = _REPO_ROOT / "evaluation" / "datasets" / "knowledge_base"
DEFAULT_DATASET = _REPO_ROOT / "evaluation" / "datasets" / "retrieval_eval.jsonl"


def build_index(knowledge_base: Path) -> list[dict[str, Any]]:
    """Chunk every file in ``knowledge_base`` and return one record per chunk.

    The record shape is deliberately the same across all three output modes so a
    human reading the table and a script reading the JSON see identical data.
    """
    documents = load_documents(knowledge_base)
    if not documents:
        logger.warning("build_dataset_ids.no_documents", path=str(knowledge_base))
        return []

    chunks = chunk_documents(documents)
    return [
        {
            "external_id": chunk.external_id,
            "source": chunk.source,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "content": chunk.content,
        }
        for chunk in chunks
    ]


def print_table(records: list[dict[str, Any]], preview_chars: int) -> None:
    """Human-readable listing, one row per chunk."""
    for record in records:
        preview = " ".join(record["content"].split())[:preview_chars]
        print(
            f"{record['external_id']}  {record['source']}  "
            f"#{record['chunk_index']:>3}  {record['token_count']:>3}tok  {preview}"
        )


def summarise(records: list[dict[str, Any]]) -> dict[str, int]:
    """Chunks per source file, so the corpus size is checkable at a glance."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record["source"]] = counts.get(record["source"], 0) + 1
    for source, count in sorted(counts.items()):
        print(f"{count:>4}  {source}")
    print(f"{sum(counts.values()):>4}  TOTAL")
    return counts


def check_dataset(dataset: Path, records: list[dict[str, Any]]) -> int:
    """Report labelled ids that no longer exist in the corpus. Returns the count.

    Editing a knowledge-base file changes every id derived from it, so a stale
    label is silent until the metrics come out wrong. Running this check after
    any corpus edit turns that into a loud failure.
    """
    if not dataset.exists():
        logger.warning("build_dataset_ids.dataset_missing", path=str(dataset))
        return 0

    known = {record["external_id"] for record in records}
    missing: dict[str, int] = {}
    total = 0
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        payload = json.loads(line)
        for doc_id in payload.get("relevant_documents", []):
            if doc_id not in known:
                missing[doc_id] = missing.get(doc_id, 0) + 1

    if missing:
        print(f"FAIL  {len(missing)} stale label(s) across {total} queries:")
        for doc_id, count in sorted(missing.items(), key=lambda item: -item[1]):
            print(f"      {doc_id}  referenced {count}x")
    else:
        print(f"OK    {total} queries, every labelled id exists in the corpus")
    return len(missing)


def main() -> int:
    """Parse arguments, emit the requested view, return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=DEFAULT_KNOWLEDGE_BASE,
        help="Directory of .md files to chunk (default: evaluation/datasets/knowledge_base)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Only list chunks whose source ends with this string",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Labelled dataset to validate against the chunk ids",
    )
    parser.add_argument(
        "--check-dataset",
        action="store_true",
        help="Validate retrieval_eval.jsonl ids and exit non-zero on stale ones",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a table",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print chunk counts per source file",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=80,
        help="Characters of chunk text to show in table mode (default: 80)",
    )
    args = parser.parse_args()

    records = build_index(args.knowledge_base)
    if args.source:
        needle = args.source.lower()
        records = [r for r in records if needle in r["source"].lower()]

    if args.json:
        print(json.dumps(records, indent=2, ensure_ascii=False))
    else:
        print_table(records, args.preview_chars)
        if args.summary or not args.source:
            print()
            summarise(records)

    stale = 0
    if args.check_dataset:
        print()
        stale = check_dataset(args.dataset, build_index(args.knowledge_base))

    logger.info(
        "build_dataset_ids.done",
        knowledge_base=str(args.knowledge_base),
        chunks=len(records),
        stale_labels=stale,
    )
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())

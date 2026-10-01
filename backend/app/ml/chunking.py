"""Document loading and recursive character chunking.

Chunking is the cheapest possible knob to reason about and the one that most
often ruins retrieval, so this module is deliberately explicit rather than
clever:

* splits on the coarsest boundary that fits (paragraph, then sentence, then
  word, then character as a last resort) so a chunk ends where a human would
  end a passage;
* never cuts mid-word, except when a single word is longer than ``chunk_size``,
  where there is no alternative;
* stamps every chunk with a **content-derived** ``external_id`` so re-ingesting
  an unchanged file produces the same ids. That idempotency is what lets the
  index be rebuilt from scratch without orphaning rows.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from app.config import settings
from app.core.logging import get_logger
from app.utils.tokens import estimate_tokens

logger = get_logger(__name__)

__all__ = [
    "Chunk",
    "RawDocument",
    "chunk_documents",
    "chunk_text",
    "load_documents",
    "relative_source",
    "split_recursive",
]

# Separators are tried in order. ``""`` is the last resort: it means "split
# every character", which is the only thing that always makes progress.
_SEPARATORS: Final[tuple[str, ...]] = ("\n\n", "\n", ". ", "; ", ", ", " ", "")

# A heading on its own line is a natural chunk boundary in Markdown, and the
# generated knowledge base leans heavily on headings.
_HEADING_RE: Final = re.compile(r"^#{1,6}\s+", re.MULTILINE)

# Recognised source extensions. Anything else in the knowledge base directory is
# skipped with a log line rather than fed to the splitter as binary noise.
_SUPPORTED_SUFFIXES: Final[frozenset[str]] = frozenset({".md", ".txt"})

# Windows-safe joiner for the recursive splitter: a separator must survive the
# strip() that each level performs, otherwise ``"\n\n"`` collapses to nothing and
# text gets glued together.
_JOINER: Final[str] = " "


@dataclass(frozen=True, slots=True)
class RawDocument:
    """A source file loaded from disk, before chunking.

    ``content_hash`` is the sha256 of the file's text. It is what makes chunk
    ids stable across re-ingests and what lets the service skip re-embedding a
    file nobody has edited.
    """

    source: str
    title: str
    content: str
    content_hash: str
    path: Path | None = field(default=None, compare=False)

    @property
    def token_count(self) -> int:
        """Deterministic token estimate for the whole file."""
        return estimate_tokens(self.content)


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit: a slice of a document with a stable identity.

    ``external_id`` is what every other layer (BM25, FAISS, retrieved-document
    telemetry) keys on, so it must be a pure function of *where the chunk came
    from and which chunk it is* — never of the model, the index, or the run.
    """

    external_id: str
    source: str
    title: str
    content: str
    chunk_index: int
    token_count: int

    def as_document_tuple(self) -> tuple[str, str]:
        """``(id, text)`` pair, the shape BM25Index and the reranker expect."""
        return self.external_id, self.content


def relative_source(path: Path, base: Path) -> str:
    """Path of ``path`` relative to the knowledge-base ``base``, with ``/`` slashes.

    Forward slashes are forced so an index built on Windows and one built on
    Linux produce the *same* ``external_id`` for the same document. Without this
    the whole chunk-id scheme would silently fork per operating system.
    """
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        # Outside the base directory (a caller passed an explicit file list):
        # fall back to the file name, still posix, rather than leaking an
        # absolute path into a persisted identifier.
        return path.name


def _split_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive split over :data:`_SEPARATORS`, keeping ``overlap`` characters.

    The recursion works because the "words" that do not fit are themselves
    recursively split by the next-finer separator, until a single character is
    reached. That is what guarantees the function always terminates: there is no
    input for which a piece is longer than ``chunk_size`` on the last pass.
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    for separator in _SEPARATORS:
        if separator == "":
            return _hard_wrap(text, chunk_size, overlap)

        if separator not in text:
            continue

        # keepends=True: the whitespace *is* the boundary, and dropping it here
        # would concatenate the two words on either side into one token.
        #
        # The separator's own length counts toward the budget. A space is one
        # character, so a naive fit test lets each piece run one character over
        # -- and a piece that overruns is pushed down to the next separator,
        # where the *same* mistake repeats and it ends up in the final hard-wrap
        # branch. That is how "never cuts mid-word" quietly became false: the
        # recursive call was reached with an over-long piece that had no
        # separator left to split on, so it got character-cut. Measuring the
        # join honestly keeps every piece within budget on the first pass.
        parts = text.split(separator)
        merged: list[str] = []
        current = ""
        for part in parts:
            candidate = (
                f"{current}{separator}{part}" if current else part
            )
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                merged.append(current)
            # A part that is over budget on its own has no separator inside it;
            # it goes straight to the hard wrap rather than through another
            # level of this loop, which would find nothing to split on.
            if len(part) > chunk_size:
                merged.append(part)
                current = ""
            else:
                # The oversized part is pushed down to the next separator.
                current = part

        if current:
            merged.append(current)

        refined: list[str] = []
        for piece in merged:
            if len(piece) <= chunk_size:
                refined.append(piece)
            else:
                # Separator-free by construction: a part over budget on its own
                # means no separator of this kind occurs inside it, and every
                # finer separator was already tried. Only _hard_wrap can make
                # progress, and only it can legitimately cut mid-word.
                refined.extend(_hard_wrap(piece, chunk_size, overlap))

        if refined:
            return _apply_overlap(refined, overlap, chunk_size)

    return _hard_wrap(text, chunk_size, overlap)


def _hard_wrap(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Last-resort fixed-width split for a run with no usable separator.

    Only reached when a single token is wider than ``chunk_size`` (a URL, a
    base64 blob, a run of CJK with no spaces). There is no word boundary left
    to respect, so a hard cut is the honest behaviour.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    step = max(1, chunk_size - max(0, overlap))
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def _apply_overlap(pieces: list[str], overlap: int, chunk_size: int) -> list[str]:
    """Re-flow ``pieces`` into chunks that each carry ``overlap`` characters backward.

    Overlap exists so a sentence straddling a boundary is still wholly present
    in at least one chunk. Without it the retriever is blind to whatever the
    split happened to cut through.

    This is a **sliding window over the concatenation of the pieces**, not a
    rewrite of each one, and the distinction is the whole reason it is not the
    obvious "prepend the tail, truncate the piece". Pieces arrive from
    :func:`_split_recursive` already at the full ``chunk_size``, so a tail has
    nowhere to come from. Both obvious reactions lose text -- truncating the
    piece pushes the displaced characters out of the window rather than
    forward, and dropping the piece to make room loses it outright. Either way
    the text falls off the end of one piece and only returns if a *later* piece
    still contains it, which for a run of full pieces never happens: the
    document loses its tail, word by word, until only the first chunk is intact.

    Walking the window is what makes the properties hold, and each one costs an
    explicit check:

    * **Nothing is dropped.** Each window covers ``text[start:end]`` and the
      next ``start`` is never beyond ``end``, so the windows tile the document.
    * **Nothing exceeds** ``chunk_size``: the end is a *floor* on ``start +
      chunk_size``, not a ceiling, and is only ever pulled back.
    * **No boundary lands mid-word.** Both ends are snapped to whitespace, so a
      chunk always starts on a word and ends with one whole.
    * **The last window is not special.** It takes ``text[start:]`` outright, so
      the end of the document cannot be lost the way it was when the final
      piece was truncated to make room for its predecessor's tail.

    The residual cost is that snapping forward means the *effective* overlap is
    a little wider than requested. That is the right direction to err: an
    overlap that lands mid-word is not overlap at all.
    """
    if overlap <= 0 or len(pieces) < 2:
        return [piece for piece in pieces if piece]

    text = _JOINER.join(piece for piece in pieces if piece)
    length = len(text)
    if not length:
        return []

    chunks: list[str] = []
    start = 0
    while start < length:
        end = min(start + chunk_size, length)
        if end >= length:
            chunks.append(text[start:])
            break

        # Pull the end back onto a word boundary. ``rfind`` searches up to but
        # not including ``end``, so a space exactly at ``end`` is not a candidate.
        boundary = text.rfind(_JOINER, start, end)
        if boundary <= start:
            # No whitespace anywhere in the window: one unbroken token, longer
            # than a chunk. There is no boundary to stop on and none to resume
            # from, so emit it whole and restart after it. Splitting mid-token
            # here would be the one place that is unavoidable, and it still
            # loses nothing.
            chunks.append(text[start:end])
            start = end
            continue

        chunks.append(text[start:boundary])
        # Resume ``overlap`` characters back so the next window opens on text
        # this one already showed the reader, then walk forward to the next
        # word so the new chunk does not begin mid-word. ``start + 1`` keeps the
        # loop moving when the overlap is wider than the chunk; it cannot skip a
        # word, because the space at ``boundary`` is always still ahead of it.
        resume = max(start + 1, boundary - overlap)
        resume = text.find(_JOINER, resume) + 1  # find() cannot fail: see below
        start = resume if resume <= boundary else boundary
    return chunks


def _trim_head(text: str, budget: int) -> str:
    """The first ``budget`` characters of ``text``, cut at a word boundary.

    .. deprecated::
        Unused since ``_apply_overlap`` was rewritten to slide a window over
        the joined text. The overlap is now produced by the window's own start
        position rather than by prepending a trimmed tail to the next piece,
        so no per-piece trimming is needed on either end. Kept only as a
        reference for the word-boundary rule; delete it if nothing imports it.
    """
    if budget <= 0 or not text:
        return ""
    if len(text) <= budget:
        return text
    head = text[:budget]
    cut = head.rfind(" ")
    return head[:cut] if cut > 0 else head


def _trim_tail(text: str, overlap: int) -> str:
    """The last ``overlap`` characters of ``text``, cut at a word boundary.

    .. deprecated::
        Unused for the same reason as :func:`_trim_head`. The overlap window
        in ``_apply_overlap`` already lands on a boundary at both ends, so
        there is nothing left to trim.
    """
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    if len(text) <= overlap:
        return text
    cut = tail.find(" ")
    return tail[cut + 1 :] if cut >= 0 else tail


def split_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Public wrapper around the internal splitter.

    Normalises the input first: newlines are unified, runs of blank lines are
    collapsed to a single paragraph break, and every piece is stripped. Doing
    that up front is what makes "prefers paragraph boundaries" true in practice
    rather than only for well-formatted Markdown.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text or not text.strip():
        return []

    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    normalised = re.sub(r"\n{3,}", "\n\n", normalised)
    normalised = normalised.strip()
    if not normalised:
        return []

    pieces = _split_recursive(normalised, chunk_size, overlap)
    return [piece.strip() for piece in pieces if piece.strip()]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split ``text`` into overlapping chunks on natural language boundaries.

    Defaults come from ``settings.chunk_size`` / ``settings.chunk_overlap`` so
    the ingestion config and the retrieval config can never drift apart.

    Returns ``[]`` for empty or whitespace-only input rather than a single empty
    string, because an empty chunk would be indexed and then retrieve as a
    zero-length document.
    """
    size = chunk_size if chunk_size is not None else settings.chunk_size
    lap = overlap if overlap is not None else settings.chunk_overlap
    return split_recursive(text, size, lap)


def load_documents(path: str | Path) -> list[RawDocument]:
    """Read every ``.md`` / ``.txt`` file under ``path`` into a :class:`RawDocument`.

    ``path`` may be a single file or a directory tree; directories are walked
    recursively and results are sorted by source so ingestion order (and
    therefore chunk ordering) is reproducible across machines. Unreadable or
    unsupported files are logged and skipped — one bad file in a knowledge base
    must not abort a whole ingestion run.
    """
    root = Path(path)
    if not root.exists():
        logger.warning("ml.chunking.path_missing", path=str(root))
        return []

    if root.is_file():
        candidates = [root]
        base = root.parent
    else:
        base = root
        candidates = sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
        )

    documents: list[RawDocument] = []
    for file_path in candidates:
        document = _load_one(file_path, base)
        if document is not None:
            documents.append(document)

    logger.info(
        "ml.chunking.documents_loaded",
        path=str(root),
        count=len(documents),
        total_chars=sum(len(d.content) for d in documents),
    )
    return documents


def _load_one(file_path: Path, base: Path) -> RawDocument | None:
    """Read and hash a single file, or ``None`` when it cannot be used."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "ml.chunking.read_failed",
            path=str(file_path),
            error=str(exc),
        )
        return None

    if not text.strip():
        logger.debug("ml.chunking.empty_document", source=str(file_path))
        return None

    source = relative_source(file_path, base)
    return RawDocument(
        source=source,
        title=derive_title(text, source),
        content=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        path=file_path,
    )


def derive_title(text: str, fallback: str) -> str:
    """Best available human title for a document.

    First Markdown H1 wins, else the first non-empty line, else the file name.
    The title rides on every chunk so a retrieved passage in the UI has a label
    a human recognises without opening the file.
    """
    match = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:200]
    return Path(fallback).stem or fallback


def make_external_id(source: str, content_hash: str, chunk_index: int) -> str:
    """Content-addressed id for one chunk: ``sha256(source|hash|index)[:16]``.

    Including the file hash means an edited document produces entirely new ids,
    which is what lets the caller delete stale chunks by set difference instead
    of by heuristics. Truncated to 16 hex chars (64 bits) because these ids
    travel through JSON APIs, BM25 maps and FAISS id lists; a longer digest buys
    nothing at that scale.
    """
    payload = f"{source}|{content_hash}|{chunk_index}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def chunk_documents(
    documents: list[RawDocument],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Chunk every document, stamping stable ids and deterministic token counts.

    Token counts come from :func:`app.utils.tokens.estimate_tokens` — never from
    an LLM and never from a model download — because a token count is a number
    the dashboard will later aggregate, and it has to be the same number on the
    SDK, the collector and the server.
    """
    size = chunk_size if chunk_size is not None else settings.chunk_size
    lap = overlap if overlap is not None else settings.chunk_overlap

    chunks: list[Chunk] = []
    for document in documents:
        for index, piece in enumerate(chunk_text(document.content, size, lap)):
            chunks.append(
                Chunk(
                    external_id=make_external_id(
                        document.source, document.content_hash, index
                    ),
                    source=document.source,
                    title=document.title,
                    content=piece,
                    chunk_index=index,
                    token_count=estimate_tokens(piece),
                )
            )

    logger.info(
        "ml.chunking.chunks_created",
        documents=len(documents),
        chunks=len(chunks),
        chunk_size=size,
        overlap=lap,
    )
    return chunks

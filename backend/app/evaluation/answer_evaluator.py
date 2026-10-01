"""Deterministic answer evaluation, with an optional and quarantined LLM judge.

Everything in this module that produces a number used by the dashboard is
deterministic: claim extraction is sentence splitting, support is content-word
alignment, and the two similarity-based scores are cosine similarity over real
embedding vectors. No LLM is involved in any of it, because a metric that moves
when the model moves cannot be used to detect that something else regressed.

The one place a model appears is :class:`LLMJudge`, and it is behind three
separate locks: ``settings.llm_judge_enabled`` (off by default), an explicit
opt-in flag on the call, and an exception guard that converts every failure into
``None``. Its scores land in ``judge_faithfulness`` / ``judge_answer_relevance``
and *nowhere else*.

**Do not blindly trust an LLM judge.** A judge is a second opinion, not a
measurement: it is not reproducible across model versions, it cannot be
hand-verified against a hand-computed example, and it silently absorbs every
bias the graded model already has. So the deterministic score is the one that is
stored, compared and regressed against, and a judged value is only ever written
into its own ``judge_*`` column. Overwriting a deterministic number with a
judged one would mean the dashboard reported a number that no check on the
implementation could ever reproduce — which is precisely the failure this
project is built to rule out.

Embeddings are injected, never assumed. ``evaluate_context_relevance`` and
``evaluate_answer_relevance`` are *defined as* cosine similarity, so there is
no honest fallback when no embedder is available: they raise
:class:`EmbedderRequiredError` rather than returning a plausible number.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.config import settings
from app.core.logging import get_logger
from app.schemas.evaluation import AnswerEvaluationResult, ClaimEvaluation

logger = get_logger(__name__)

__all__ = [
    "ANSWER_RELEVANCE_WEIGHTS",
    "CitationCoverageResult",
    "ContextRelevanceResult",
    "EmbedderLike",
    "EmbedderRequiredError",
    "GroundingResult",
    "JUDGED_BY",
    "JudgeScores",
    "LLMJudge",
    "ScoredDocument",
    "AnswerEvaluationReport",
    "citation_coverage",
    "citation_report",
    "content_word_coverage",
    "evaluate_answer",
    "evaluate_answer_detailed",
    "evaluate_answer_relevance",
    "evaluate_context_relevance",
    "extract_claims",
    "grounding_score",
    "normalise_documents",
    "parse_citation_markers",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Recorded on every result this module produces, so a stored number always
#: says how it was computed. Matches the vocabulary in
#: ``app.schemas.evaluation.AnswerEvaluationResult.method``.
DETERMINISTIC_METHOD = "deterministic_embedding"
#: ``method`` for grounding that ran on word alignment only, because no embedder
#: was supplied. A weaker method is labelled, never silently substituted.
ALIGNMENT_METHOD = "deterministic_alignment"

#: The only value ``judged_by`` is ever allowed to take.
JUDGED_BY = "llm_judge"

#: Weighting of the two components of answer relevance: embedding similarity and
#: question-content-word coverage. Fixed, documented and stored alongside the
#: result rather than hidden in a formula — a blended score whose weights nobody
#: can read is indistinguishable from a fabricated one. Both components are also
#: reported individually on :class:`ContextRelevanceResult` / the report.
ANSWER_RELEVANCE_WEIGHTS: dict[str, float] = {"similarity": 0.7, "coverage": 0.3}

#: A claim whose content words are all present in one context sentence is
#: supported regardless of cosine similarity. A paraphrase scores low on cosine
#: while being fully entailed, and calling that "unsupported" would be the
#: deterministic scorer inventing a fact about the answer.
CLAIM_OVERLAP_THRESHOLD = 0.8
#: Weakest content-word coverage still worth counting as partial support.
CLAIM_PARTIAL_OVERLAP_THRESHOLD = 0.5

#: Small English function words carrying no retrieval signal. Deliberately a
#: fixed literal: pulling a stopword list from a package at import time would
#: make the score depend on a dependency's version.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from
    by with without within into over under about as is are was were be been being am
    do does did done have has had having will would shall should can could may might
    must it its it's they them their there here what which who whom whose when where
    why how all any both each few more most other some such no nor not only own same
    so too very s t don now i you he she we us our your my me him her
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")

# Citation markers. Three shapes, one alternation, so a marker can never be
# matched twice and the reported denominator is well defined:
#   ``[3]`` / ``[S3]`` / ``[doc-2]``  -> the bracketed branch
#   ``[source: billing faq]``         -> the keyed bracketed branch
#   ``(source: billing faq)``         -> the parenthetical branch
_NUMERIC_TOKEN_RE = re.compile(r"^[A-Za-z]*(\d+)$")

# The gap must be a comma (or semicolon) or a dash, *not* an optional space.
# ``[1,3]`` is a marker list; ``[1 3]`` is two adjacent references. Allowing a
# bare space between two tokens lets ``[see 3]`` parse as the token "see" and
# the token "3", counting a word as a citation and diluting coverage.
_MARKER_RE = re.compile(
    r"\[(?:(?:source|src|ref|doc|document)\s*[:=]\s*(?P<keyed>[^\]\r\n]+?)\s*)\]"
    r"|\[\s*(?P<token>[A-Za-z0-9][A-Za-z0-9_.\-]*\s*(?:[,;/]\s*[A-Za-z0-9][A-Za-z0-9_.\-]*\s*)*)\]"
    r"|\(\s*(?:source|src|ref|doc|document)\s*[:=]\s*(?P<paren>[^)\r\n]+?)\s*\)",
    re.IGNORECASE,
)
_MARKER_LIST_SPLIT_RE = re.compile(r"[,;/]")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[ \t]+|\n+")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EmbedderRequiredError(RuntimeError):
    """Raised when a similarity metric is asked for without an embedder.

    Both similarity metrics are *defined* as a cosine over real vectors, so
    there is no degraded mode to fall back to. Returning a constant, a zero, or
    a score computed some other way would put a number on the dashboard that no
    implementation of the metric ever produced.
    """


# ---------------------------------------------------------------------------
# Embedder protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbedderLike(Protocol):
    """The slice of :class:`app.ml.embeddings.Embedder` this module needs.

    A Protocol rather than the concrete class so a test can inject a tiny fake
    with hand-written vectors and assert the cosine arithmetic against values
    computed by hand. Loading a 90 MB checkpoint to check that ``0.6 * 5.0`` is
    ``3.0`` would be absurd.
    """

    @property
    def model_name(self) -> str:
        """Identifier of the checkpoint, recorded on every result."""
        ...

    def embed(self, texts: list[str]) -> Any:
        """Return an ``(len(texts), dimension)`` matrix of vectors."""
        ...


def _require_embedder(embedder: EmbedderLike | None, metric: str) -> EmbedderLike:
    if embedder is None:
        raise EmbedderRequiredError(
            f"{metric} is defined as cosine similarity over embedding vectors and "
            "cannot be computed without one. Pass embedder=... "
            "(app.ml.embeddings.get_embedder() for the configured model, or any "
            "object with an .embed(list[str]) method). Returning a placeholder "
            "here would put an unmeasured number on the dashboard."
        )
    return embedder


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    """One retrieved passage, normalised from whatever the caller passed.

    ``id`` is the identity used for citation resolution, ``text`` the content
    that is embedded, and ``index`` the 0-based position the passage occupied in
    the retriever's ranking. Position matters: ``[1]`` in an answer is, by
    long-standing convention, the first passage, so a marker that matches no id
    is still resolved when it names a real position.
    """

    id: str
    text: str
    index: int
    similarity: float | None = None


@dataclass(frozen=True, slots=True)
class ContextRelevanceResult:
    """How well the retrieved context answers the question that was asked."""

    score: float
    similarities: list[float]
    best_index: int | None
    best_similarity: float | None
    num_documents: int
    embedding_model: str | None
    method: str = DETERMINISTIC_METHOD
    judged_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "similarities": list(self.similarities),
            "best_index": self.best_index,
            "best_similarity": self.best_similarity,
            "num_documents": self.num_documents,
            "embedding_model": self.embedding_model,
            "method": self.method,
            "judged_by": self.judged_by,
        }


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """Claim-level support for an answer, and the claims that had none.

    ``score`` is supported claims / total claims. ``unsupported_claim_ratio`` is
    the same measurement inverted; both are stored because the dashboard reports
    them in different places and deriving one from the other at read time is a
    step where a rounding bug could hide.
    """

    score: float
    claims: list[ClaimEvaluation]
    unsupported_claim_ratio: float
    num_claims: int
    method: str
    embedding_model: str | None = None
    judged_by: str | None = None

    @property
    def supported_claims(self) -> list[ClaimEvaluation]:
        return [claim for claim in self.claims if claim.supported]


@dataclass(frozen=True, slots=True)
class CitationCoverageResult:
    """Citation markers found in an answer, and how many resolved.

    Carries the raw counts because ``coverage == 1.0`` is ambiguous on its own:
    it means either "every citation resolves" or "there was nothing to resolve",
    and those are very different answers about the same generated text.
    """

    coverage: float
    cited: int
    resolved: int
    markers: list[str]
    unresolved: list[str]

    @property
    def has_citations(self) -> bool:
        return self.cited > 0


@dataclass(frozen=True, slots=True)
class JudgeScores:
    """A second opinion from the local model, quarantined from the real metrics.

    Every field here maps to a ``judge_*`` column and nothing else. The
    constructor is not public API, but the type exists so that no caller can
    accidentally splat a ``JudgeScores`` into the deterministic columns.
    """

    faithfulness: float | None = None
    answer_relevance: float | None = None
    rationale: str | None = None
    model: str | None = None
    judged_by: str = JUDGED_BY
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AnswerEvaluationReport:
    """Everything one answer evaluation produced, including its provenance.

    :meth:`to_schema` projects onto the API shape so a route can serialise
    without knowing that a second, richer type exists.
    """

    result: AnswerEvaluationResult
    judged_by: dict[str, str] = field(default_factory=dict)
    context_relevance: ContextRelevanceResult | None = None
    grounding: GroundingResult | None = None
    citations: CitationCoverageResult | None = None
    answer_relevance_components: dict[str, float | None] = field(default_factory=dict)

    def to_schema(self) -> AnswerEvaluationResult:
        """The API-facing projection, judge columns kept separate."""
        return self.result


# ---------------------------------------------------------------------------
# Document normalisation
# ---------------------------------------------------------------------------


def _document_id(source: Any, fallback: str) -> str:
    """Best-effort identity for a passage, from mapping, ORM object or plain str."""
    if isinstance(source, str):
        return fallback
    if isinstance(source, Mapping):
        for key in ("id", "document_id", "external_id", "source"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return fallback
    for key in ("id", "document_id", "external_id", "source"):
        value = getattr(source, key, None)
        if value is not None and str(value).strip():
            return str(value)
    return fallback


def _document_text(source: Any) -> str:
    """The embeddable content of a passage."""
    if isinstance(source, str):
        return source
    if isinstance(source, Mapping):
        for key in ("content", "text", "preview", "title"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    for key in ("content", "text", "content_preview"):
        value = getattr(source, key, None)
        if isinstance(value, str) and value.strip():
            return value
    title = getattr(source, "title", None)
    return title if isinstance(title, str) else ""


def normalise_documents(
    documents: Sequence[Any],
) -> list[ScoredDocument]:
    """Coerce any accepted passage representation into :class:`ScoredDocument`.

    Callers pass retrieved chunks, ORM ``Document`` rows, ingested dictionaries
    or bare strings depending on where they are in the pipeline. Normalising once
    here means the similarity, grounding and citation code all read the same two
    fields, and a shape mismatch surfaces once instead of as a silent zero.
    """
    normalised: list[ScoredDocument] = []
    for position, source in enumerate(documents or []):
        text = _document_text(source)
        identifier = _document_id(source, fallback=str(position))
        if identifier is None:  # pragma: no cover - _document_id never returns None
            identifier = str(position)
        normalised.append(ScoredDocument(id=identifier, text=text, index=position))
    return normalised


# ---------------------------------------------------------------------------
# Tokenisation helpers (deterministic, no model)
# ---------------------------------------------------------------------------


def _content_words(text: str) -> set[str]:
    """Lowercase word set with function words removed.

    Used for both directions of alignment: claim against context, and question
    against answer. Dropping stopwords is what keeps "does the" from making an
    unrelated sentence look like a match.
    """
    return {word for word in _WORD_RE.findall(text.lower()) if word not in _STOPWORDS}


def _overlap(needle: set[str], haystack: set[str]) -> float:
    """Fraction of ``needle`` present in ``haystack``; ``0.0`` when empty.

    Directional on purpose. For grounding, the claim is the subject and the
    context sentence is the evidence, so every content word of the claim must be
    accounted for. Symmetric overlap would let a long context sentence "support"
    a short claim just by being long.
    """
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)


def content_word_coverage(question: str, answer: str) -> float:
    """Fraction of the question's content words that appear in the answer.

    The non-embedding half of answer relevance. A paraphrase of a question can
    answer it perfectly while sharing almost no words with it, which is why this
    is blended with cosine similarity rather than used alone.
    """
    return _overlap(_content_words(question), _content_words(answer))


def extract_claims(answer: str) -> list[str]:
    """Split an answer into claim-sized sentences, deterministically.

    A claim is the unit support is measured over. Sentence splitting is a plain
    regex rather than an NLP model for the same reason everything else here is:
    the split has to be reproducible, and a model would make it vary between runs
    of the same answer.
    """
    if not answer or not answer.strip():
        return []
    claims: list[str] = []
    for raw in _SENTENCE_SPLIT_RE.split(answer):
        cleaned = raw.strip()
        if len(cleaned) >= 2:
            claims.append(cleaned)
    return claims


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _embed_rows(embedder: EmbedderLike, texts: list[str]) -> Any:
    """Embed and L2-normalise so the dot product *is* the cosine.

    Normalisation happens here rather than trusting the checkpoint: the store's
    FAISS index assumes normalised vectors, but a caller may inject an embedder
    that does not, and an unnormalised vector would silently produce a similarity
    above 1.0 that is then clamped into looking plausible.
    """
    import numpy as np  # local: keeps module import free of numpy cost

    matrix = np.asarray(embedder.embed(texts), dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[0] != len(texts):
        raise ValueError(
            f"embedder returned {matrix.shape[0]} vectors for {len(texts)} texts; "
            "embeddings must be one row per input, in order"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector has no direction; leaving it as zeros makes its similarity a
    # defined 0.0 instead of a NaN that poisons a mean.
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _cosines(reference: Any, matrix: Any) -> list[float]:
    """Cosine of every row of ``matrix`` against a single normalised reference."""
    return [float(value) for value in (matrix @ reference)]


def _clamp(value: float) -> float:
    """Clamp to ``[0.0, 1.0]``; a similarity of 1.0000001 is arithmetic noise."""
    return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------


def evaluate_context_relevance(
    question: str,
    documents: Sequence[Any],
    *,
    embedder: EmbedderLike | None = None,
) -> ContextRelevanceResult:
    """Mean cosine similarity between the question and each retrieved passage.

    The score is the mean over *all* supplied passages, including irrelevant
    ones — a retriever that returned five passages when two were relevant should
    not score well for it. ``similarities`` and ``best_index`` are kept so a low
    score can be diagnosed (nothing relevant at all, versus one good passage
    buried at rank 5).

    Raises :class:`EmbedderRequiredError` without an embedder, and ``ValueError``
    when the question is empty or no passages were supplied — there is no
    meaningful mean over an empty set, and returning 0.0 would read as "the
    retrieved context was completely irrelevant".
    """
    active = _require_embedder(embedder, "evaluate_context_relevance")
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    passages = normalise_documents(documents)
    if not passages:
        raise ValueError("evaluate_context_relevance requires at least one document")

    texts = [passage.text or "" for passage in passages]
    matrix = _embed_rows(active, [question, *texts])
    question_vector, passage_vectors = matrix[0], matrix[1:]
    similarities = _cosines(question_vector, passage_vectors)

    best_position = max(range(len(similarities)), key=lambda i: similarities[i])
    return ContextRelevanceResult(
        score=_clamp(sum(similarities) / len(similarities)),
        similarities=[_clamp(value) for value in similarities],
        best_index=passages[best_position].index,
        best_similarity=_clamp(similarities[best_position]),
        num_documents=len(passages),
        embedding_model=getattr(active, "model_name", None),
    )


def evaluate_answer_relevance(
    question: str,
    answer: str,
    *,
    embedder: EmbedderLike | None = None,
) -> float:
    """Cosine similarity between the question and the generated answer.

    Pure embedding similarity. The blended question-coverage form used by
    :func:`evaluate_answer` lives in
    :data:`ANSWER_RELEVANCE_WEIGHTS` so the weights are inspectable; this
    function stays the single unblended measurement.

    Raises :class:`EmbedderRequiredError` without an embedder.
    """
    active = _require_embedder(embedder, "evaluate_answer_relevance")
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not answer or not answer.strip():
        raise ValueError("answer must be a non-empty string")

    matrix = _embed_rows(active, [question, answer])
    return _clamp(float(matrix[1] @ matrix[0]))


# ---------------------------------------------------------------------------
# Citation markers
# ---------------------------------------------------------------------------


def _marker_token(raw: str) -> str:
    """Normalise a marker to the identifier it names.

    ``S1`` and ``src_1`` both name ``1``; the leading alphabetic tag is a
    namespace the corpus chose, not part of the identifier.
    """
    token = raw.strip().strip(".,;:")
    numeric = _NUMERIC_TOKEN_RE.match(token)
    return numeric.group(1) if numeric else token


def parse_citation_markers(answer: str) -> list[str]:
    """Every citation marker in ``answer``, in order of appearance.

    Duplicates are kept. The coverage denominator is *citations made*, so an
    answer that cites the same unresolvable source twice has two unresolvable
    citations, not one; collapsing duplicates first would let repetition hide a
    bad citation.
    """
    if not answer:
        return []
    markers: list[str] = []
    for match in _MARKER_RE.finditer(answer):
        for group in ("keyed", "token", "paren"):
            value = match.group(group)
            if value:
                # A bracket can hold a *list* -- ``[1,3]`` cites two sources.
                # Splitting keeps the reported denominator equal to the number
                # of citations actually made, which is what coverage is over.
                if group == "token":
                    markers.extend(
                        _marker_token(part)
                        for part in _MARKER_LIST_SPLIT_RE.split(value)
                        if part.strip()
                    )
                else:
                    markers.append(_marker_token(value))
                break
    return markers


def _document_aliases(passage: ScoredDocument, source: Any) -> set[str]:
    """Every string a citation might legitimately use for this passage.

    Includes the id, the 1-based position (the ``[1]`` convention) and, when the
    caller supplied one, the title and source path — people cite documents by
    name far more often than by hash.
    """
    aliases = {passage.id, str(passage.index + 1)}
    if isinstance(source, Mapping):
        for key in ("title", "source"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                aliases.add(value.strip())
    else:
        for key in ("title", "source"):
            value = getattr(source, key, None)
            if isinstance(value, str) and value.strip():
                aliases.add(value.strip())
    return {alias.strip().casefold() for alias in aliases}


def citation_report(answer: str, documents: Sequence[Any]) -> CitationCoverageResult:
    """Which citation markers in ``answer`` resolve to a supplied passage.

    A marker resolves when it matches a passage's id, its 1-based position, or
    its title/source. Matching both ways is deliberate: ``[1]`` and ``[S1]`` are
    both common in generated answers, and refusing the positional reading would
    report a fully cited answer as uncited.

    See :func:`citation_coverage` for what an answer with no markers returns.
    """
    raw_documents = list(documents or [])
    passages = normalise_documents(raw_documents)
    alias_sets = [
        _document_aliases(passage, raw_documents[position])
        for position, passage in enumerate(passages)
    ]

    markers = parse_citation_markers(answer)
    unresolved = [
        marker for marker in markers if not any(marker.casefold() in a for a in alias_sets)
    ]
    resolved = len(markers) - len(unresolved)
    return CitationCoverageResult(
        coverage=(resolved / len(markers)) if markers else 0.0,
        cited=len(markers),
        resolved=resolved,
        markers=markers,
        unresolved=unresolved,
    )


def citation_coverage(answer: str, documents: Sequence[Any]) -> float:
    """Fraction of an answer's citation markers that resolve to a supplied document.

    Deterministic string parsing — no model, no network, no threshold.

    Two edges are worth stating, because both are choices rather than
    consequences:

    * An answer with **no** markers scores ``0.0``. A vacuous "every citation
      resolved" reading would score 1.0, which puts a perfect number on an
      answer that ignored its sources entirely — the one thing citation coverage
      exists to catch. The gap between *cited nothing* and *cited wrongly* is
      still visible: both score 0.0, but :func:`citation_report` reports
      ``cited=0`` for the first, and that is the field to aggregate on if you
      need to tell them apart.
    * A marker that resolves to nothing is **counted against** coverage rather
      than ignored. Ignoring it would make hallucinated citations free.
    """
    return citation_report(answer, documents).coverage


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def _claim_sentences(context: Sequence[ScoredDocument]) -> list[tuple[int, str, set[str]]]:
    """Every context sentence as ``(passage_index, sentence, content_words)``.

    Support is measured per *sentence*, not per passage: a claim can be entailed
    by one line of a long passage and contradicted by the next, and scoring the
    passage as a whole would blur that into an average.
    """
    sentences: list[tuple[int, str, set[str]]] = []
    for passage in context:
        for sentence in extract_claims(passage.text):
            sentences.append((passage.index, sentence, _content_words(sentence)))
    return sentences


def grounding_score(
    answer: str,
    context: Sequence[Any],
    *,
    embedder: EmbedderLike | None = None,
    similarity_threshold: float | None = None,
) -> GroundingResult:
    """Fraction of the answer's claims that the context actually supports.

    A claim is supported when either

    * its cosine similarity to some context sentence reaches
      ``settings.grounding_similarity_threshold``, or
    * a context sentence contains all of its content words
      (:data:`CLAIM_OVERLAP_THRESHOLD`), which is what catches a claim that
      paraphrases a long sentence and therefore scores poorly on cosine.

    Without an embedder the word-alignment path still runs and
    :attr:`GroundingResult.method` becomes :data:`ALIGNMENT_METHOD`. That is a
    genuine, weaker measurement — reported as itself, never presented as the
    embedding score. ``evaluate_context_relevance`` has no such fallback because
    its score *is* the similarity.
    """
    threshold = (
        settings.grounding_similarity_threshold
        if similarity_threshold is None
        else float(similarity_threshold)
    )
    claims = extract_claims(answer)
    context_passages = normalise_documents(context)
    sentences = _claim_sentences(context_passages)

    if not claims:
        # An empty answer has no unsupported claims and no supported ones. 1.0
        # with zero claims recorded keeps the invariant score = supported/total
        # vacuously true, and ``num_claims == 0`` is what actually tells the
        # reader there was nothing to measure.
        return GroundingResult(
            score=1.0,
            claims=[],
            unsupported_claim_ratio=0.0,
            num_claims=0,
            method=ALIGNMENT_METHOD if embedder is None else DETERMINISTIC_METHOD,
            embedding_model=getattr(embedder, "model_name", None) if embedder else None,
        )

    if embedder is not None and sentences:
        claim_texts = [claim for claim in claims]
        sentence_texts = [sentence for _, sentence, _ in sentences]
        matrix = _embed_rows(embedder, [*claim_texts, *sentence_texts])
        claim_vectors = matrix[: len(claim_texts)]
        sentence_vectors = matrix[len(claim_texts) :]
        similarities = claim_vectors @ sentence_vectors.T
    else:
        if embedder is not None:
            logger.warning(
                "evaluation.grounding.no_context_sentences",
                claims=len(claims),
                method=ALIGNMENT_METHOD,
            )
        similarities = None

    evaluations: list[ClaimEvaluation] = []
    supported_count = 0

    for position, claim in enumerate(claims):
        claim_words = _content_words(claim)
        best_index: int | None = None
        best_similarity = 0.0
        best_overlap = 0.0
        method = ALIGNMENT_METHOD

        for sentence_position, (_, _, sentence_words) in enumerate(sentences):
            overlap = _overlap(claim_words, sentence_words)
            similarity = (
                float(similarities[position, sentence_position])
                if similarities is not None
                else 0.0
            )
            if overlap >= CLAIM_OVERLAP_THRESHOLD:
                # Full content-word containment settles it; take the *least* of
                # the two similarities seen for that sentence so the reported
                # number reflects a gap if one exists rather than being
                # flattering.
                best_index, best_similarity, best_overlap = sentence_position, similarity, overlap
                method = DETERMINISTIC_METHOD if similarities is not None else ALIGNMENT_METHOD
                break
            if similarity > best_similarity:
                best_index, best_similarity, best_overlap = sentence_position, similarity, overlap
                method = "embedding_similarity"
            elif overlap > best_overlap:
                best_overlap = overlap
                if best_index is None:
                    best_index = sentence_position

        is_supported = best_similarity >= threshold or best_overlap >= CLAIM_PARTIAL_OVERLAP_THRESHOLD
        if is_supported:
            supported_count += 1

        evaluations.append(
            ClaimEvaluation(
                claim=claim,
                supported=is_supported,
                best_matching_context_index=best_index,
                similarity=round(_clamp(best_similarity), 6),
                method=method,
            )
        )

    num_claims = len(claims)
    unsupported = num_claims - supported_count
    return GroundingResult(
        score=supported_count / num_claims,
        claims=evaluations,
        unsupported_claim_ratio=unsupported / num_claims,
        num_claims=num_claims,
        method=DETERMINISTIC_METHOD if similarities is not None else ALIGNMENT_METHOD,
        embedding_model=getattr(embedder, "model_name", None) if embedder else None,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def evaluate_answer_detailed(
    question: str,
    answer: str,
    context: Sequence[Any] | None = None,
    *,
    embedder: EmbedderLike | None = None,
    judge: "JudgeScores | None" = None,
) -> AnswerEvaluationReport:
    """All four deterministic sub-scores for one (question, answer) pair.

    ``grounding``, ``citations`` and ``context_relevance`` degrade gracefully
    when inputs are absent — an empty context is a real situation (a retrieval
    miss) and reports ``0.0`` support with no citations rather than raising.
    ``answer_relevance`` cannot degrade, because it *is* a cosine: without an
    embedder that one sub-score is reported as ``None`` while the others remain
    measured. Every field is therefore optional on the way out, and nothing is
    substituted to fill a gap.

    A ``judge`` argument is accepted but only ever populates ``judge_*``.
    """
    passages = normalise_documents(context or [])
    grounding = grounding_score(answer, passages, embedder=embedder)

    citations = citation_report(answer, context or [])
    citation_value = citations.coverage if citations.cited else 0.0

    context_relevance: ContextRelevanceResult | None = None
    if passages and question.strip():
        if embedder is not None:
            context_relevance = evaluate_context_relevance(question, passages, embedder=embedder)
        else:
            logger.warning(
                "evaluation.answer.no_embedder",
                sub_score="context_relevance",
                num_documents=len(passages),
            )

    similarity: float | None = None
    coverage: float | None = None
    if embedder is not None and question.strip() and answer.strip():
        similarity = evaluate_answer_relevance(question, answer, embedder=embedder)
        coverage = content_word_coverage(question, answer)

    if similarity is None or coverage is None:
        answer_relevance: float | None = None
    else:
        weights = ANSWER_RELEVANCE_WEIGHTS
        blended = (
            weights["similarity"] * similarity + weights["coverage"] * coverage
        )
        answer_relevance = _clamp(blended)

    context_value = context_relevance.score if context_relevance is not None else None

    judged_by: dict[str, str] = {}
    judge_faithfulness = None
    judge_answer_relevance = None
    judge_model = None
    if judge is not None and (judge.faithfulness is not None or judge.answer_relevance is not None):
        judge_faithfulness = judge.faithfulness
        judge_answer_relevance = judge.answer_relevance
        judge_model = judge.model
        if judge.faithfulness is not None:
            judged_by["judge_faithfulness"] = judge.judged_by
        if judge.answer_relevance is not None:
            judged_by["judge_answer_relevance"] = judge.judged_by

    schema = AnswerEvaluationResult(
        question=question,
        faithfulness=grounding.score,
        context_relevance=context_value if context_value is not None else 0.0,
        answer_relevance=answer_relevance if answer_relevance is not None else 0.0,
        citation_coverage=citation_value,
        unsupported_claim_ratio=grounding.unsupported_claim_ratio,
        judge_faithfulness=judge_faithfulness,
        judge_answer_relevance=judge_answer_relevance,
        judge_model=judge_model,
        method=grounding.method,
        claims=list(grounding.claims),
    )

    return AnswerEvaluationReport(
        result=schema,
        judged_by=judged_by,
        context_relevance=context_relevance,
        grounding=grounding,
        citations=citations,
        answer_relevance_components={
            "similarity": similarity,
            "content_word_coverage": coverage,
            "similarity_weight": ANSWER_RELEVANCE_WEIGHTS["similarity"],
            "coverage_weight": ANSWER_RELEVANCE_WEIGHTS["coverage"],
        },
    )


def evaluate_answer(
    question: str,
    answer: str,
    context: Sequence[Any] | None = None,
    *,
    embedder: EmbedderLike | None = None,
    judge: "JudgeScores | None" = None,
) -> AnswerEvaluationResult:
    """The four sub-scores as one API-shaped result.

    Thin wrapper over :func:`evaluate_answer_detailed` for callers that only
    need the serialisable shape.
    """
    return evaluate_answer_detailed(
        question, answer, context, embedder=embedder, judge=judge
    ).result


# ---------------------------------------------------------------------------
# Opt-in LLM judge
# ---------------------------------------------------------------------------

#: Prompt asks for a strict object. Ollama's ``format="json"`` constrains the
#: grammar on top of this, but the parser below does not rely on either.
_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator of retrieval-augmented answers. "
    "You are given a question, the retrieved context passages, and a generated answer. "
    "Reply with a single JSON object and nothing else, using exactly these keys:\n"
    '{"faithfulness": <float 0-1>, "answer_relevance": <float 0-1>, "rationale": "<one sentence>"}\n'
    "faithfulness: how completely every claim in the answer is supported by the context.\n"
    "answer_relevance: how well the answer addresses the question that was asked.\n"
    "Judge only from the supplied context. Do not use outside knowledge. "
    "If the context does not contain the answer, score low and say so."
)


def _coerce_unit_interval(value: Any) -> float | None:
    """Read a judge field as a number in ``[0, 1]``, or ``None`` if it is not one.

    Models are asked for ``0-1`` floats but not all of them oblige. Two wrong
    answers are plausible enough to be worth handling explicitly:

    * ``8``/``85`` — a percentage, or a rating out of ten. Rescaled by 100.
    * ``1.7``/``-0.2`` — an ordinary miscount. There is no scale that makes
      these consistent with anything, so they are rejected rather than
      rescaled: dividing ``1.7`` by 100 would silently turn a judge that
      ignored its instructions into a judge reporting near-zero faithfulness.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if numeric != numeric or numeric in (float("inf"), float("-inf")):  # NaN / inf
        return None
    if numeric < 0.0:
        return None
    if numeric > 1.0:
        # Percentages only. A value between 1 and 100 with no decimal point is
        # a percentage; anything fractional in that range is a miscount.
        if numeric > 100.0 or numeric % 1.0:
            return None
        numeric = numeric / 100.0
    return max(0.0, min(1.0, numeric))


class LLMJudge:
    """Opt-in second opinion from the local Ollama model. Never raises.

    Three gates before a model is called: ``settings.llm_judge_enabled`` must be
    true, the call must pass ``enabled=True`` explicitly, and the provider must
    report itself available. Any failure — transport, refusal, unparseable JSON,
    a score outside every plausible range — is logged and returned as ``None``,
    because an evaluation that dies because the judge is down would take the
    deterministic metrics down with it.

    Remember what this class is for: corroboration. The deterministic scores are
    what the project stores and regresses against; these land in ``judge_*``
    columns and are never allowed to overwrite a measured value.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        model: str | None = None,
        max_claims: int | None = None,
    ) -> None:
        self._enabled = settings.llm_judge_enabled if enabled is None else bool(enabled)
        self._model = model or settings.llm_judge_model or settings.ollama_model
        self._max_claims = max_claims if max_claims is not None else settings.judge_max_claims

    @property
    def enabled(self) -> bool:
        """Whether the judge is configured to run at all."""
        return self._enabled

    @property
    def model(self) -> str:
        """The local model that would be used."""
        return self._model

    async def judge(
        self,
        question: str,
        answer: str,
        context: Sequence[Any] | None = None,
        *,
        enabled: bool = False,
    ) -> JudgeScores | None:
        """Return second-opinion scores, or ``None`` for any reason.

        ``enabled=True`` on the call is the request-level opt-in; it cannot
        bypass ``settings.llm_judge_enabled`` being off, because a deployment
        that disabled the judge has done so on purpose.
        """
        if not self._enabled:
            logger.debug("evaluation.judge.disabled_by_settings")
            return None
        if not enabled:
            logger.debug("evaluation.judge.not_requested")
            return None

        passages = normalise_documents(context or [])[: self._max_claims]
        if not passages:
            # No context means there is nothing to be faithful *to*; asking a
            # model to grade grounding against an empty passage list invites a
            # confident number that measures nothing.
            logger.info("evaluation.judge.no_context")
            return None

        try:
            from app.providers.registry import get_provider

            provider = get_provider()
            if not await provider.available():
                logger.warning("evaluation.judge.provider_unavailable")
                return None

            user_message = (
                "Context passages:\n"
                + "\n".join(
                    f"[{position + 1}] {passage.text[:1200]}"
                    for position, passage in enumerate(passages)
                )
                + f"\n\nQuestion: {question}\n\nAnswer: {answer}\n\nJSON:"
            )
            response = await provider.generate(
                [
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                model=self._model,
                temperature=0.0,
                format="json",
            )
        except Exception as exc:  # noqa: BLE001 - the judge must never break evaluation
            logger.warning("evaluation.judge.failed", error=str(exc), model=self._model)
            return None

        parsed = self._parse(response.text, model=response.model)
        if parsed is None:
            return JudgeScores(model=response.model, error="unparseable judge response")

        # Rebuilt rather than mutated so the model that actually produced the
        # numbers is recorded next to them.
        return JudgeScores(
            faithfulness=parsed.faithfulness,
            answer_relevance=parsed.answer_relevance,
            rationale=parsed.rationale,
            model=response.model,
            judged_by=JUDGED_BY,
        )

    @staticmethod
    def _parse(raw: str, model: str | None = None) -> JudgeScores | None:
        """Defensive parse of the judge's reply.

        Models wrap JSON in prose and fences no matter what the prompt says, so
        this tries the whole string, then the outermost brace-delimited span,
        then gives up. A reply with no usable score yields a
        :class:`JudgeScores` carrying ``error`` — the caller can record that the
        judge ran and failed, which is different from it never having run.

        ``model`` is the model that actually produced the reply. It is threaded
        through here rather than filled in afterwards so that no code path can
        return scores with no attribution attached.
        """
        text = (raw or "").strip()
        if not text:
            return None

        candidates = [text]
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            candidates.append(text[start : end + 1])

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except (ValueError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            faithfulness = _coerce_unit_interval(payload.get("faithfulness"))
            relevance = _coerce_unit_interval(payload.get("answer_relevance"))
            if faithfulness is None and relevance is None:
                continue
            rationale = payload.get("rationale")
            return JudgeScores(
                faithfulness=faithfulness,
                answer_relevance=relevance,
                rationale=str(rationale)[:2000] if rationale is not None else None,
                model=model,
                judged_by=JUDGED_BY,
            )
        return None

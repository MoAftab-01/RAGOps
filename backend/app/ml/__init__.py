"""The RAGOps retrieval and ML stack.

Importing this package is free: no model is loaded, no file is read, no
connection is opened. The embedder (~90 MB), the reranker (~90 MB) and the
corpus are all resolved on first *use*, so a process that starts up, serves a
health check and shuts down never pays for any of them.

Modules are re-exported here for convenience, but importing them is still lazy
in spirit — none of them touch the network or the filesystem at module scope.
"""

from __future__ import annotations

from app.ml.anomaly import AnomalyDetector
from app.ml.bm25 import BM25Index
from app.ml.chunking import Chunk, RawDocument, chunk_documents, chunk_text, load_documents
from app.ml.embeddings import Embedder, get_embedder
from app.ml.hybrid import HybridRetriever, RetrievedChunk, RetrieverResult
from app.ml.reranker import CrossEncoderReranker, get_reranker
from app.ml.retriever import RetrieverService, get_retrieval_service
from app.ml.vector_store import (
    FaissVectorStore,
    QdrantUnavailableError,
    QdrantVectorStore,
    VectorStore,
    get_vector_store,
)

__all__ = [
    "AnomalyDetector",
    "BM25Index",
    "Chunk",
    "CrossEncoderReranker",
    "Embedder",
    "FaissVectorStore",
    "HybridRetriever",
    "QdrantUnavailableError",
    "QdrantVectorStore",
    "RawDocument",
    "RetrievedChunk",
    "RetrieverResult",
    "RetrieverService",
    "VectorStore",
    "chunk_documents",
    "chunk_text",
    "get_embedder",
    "get_reranker",
    "get_retrieval_service",
    "get_vector_store",
    "load_documents",
]

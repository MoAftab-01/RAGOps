"""RAGOps Python SDK — observability for RAG and LLM applications.

Quick start::

    from ragops import RAGOps

    client = RAGOps(application="my-app")

    with client.trace(user_id="demo-user") as trace:
        documents = retriever.search(query)
        trace.log_retrieval(query=query, documents=documents)
        answer = llm.generate(query, documents)
        trace.log_generation(model="qwen2.5:3b", answer=answer, prompt=query)

    client.close()

The SDK is a passive observer: it never raises into the application it
instruments, and it works with no RAGOps server running (failures are logged,
not raised).
"""

from __future__ import annotations

from .client import RAGOps
from .tracer import Trace

__version__ = "0.1.0"

# The result of a one-shot ``client.log_generation(...)`` / ``log_retrieval(...)``
# call: the completed ``Trace`` it produced, so callers can read back the
# derived token counts and the trace id without re-plumbing anything.
GenerationResult = Trace

__all__ = ["RAGOps", "Trace", "GenerationResult", "__version__"]

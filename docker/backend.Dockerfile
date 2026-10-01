# ---------------------------------------------------------------------------
# RAGOps backend. CPU-only torch so the image stays laptop-buildable; swap for
# the cu124 index from requirements-ml.txt if you want GPU inference.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/backend

# curl is only here for the container healthcheck below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements-ml.txt ./

# CPU wheels for torch/faiss keep the build small; the pinned cu124 index in
# requirements-ml.txt would pull multi-GB CUDA libraries.
RUN sed -i 's|^--extra-index-url.*||' requirements-ml.txt \
    && sed -i 's|^torch==.*|torch==2.5.1+cpu|' requirements-ml.txt \
    && pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

COPY backend/ /app/backend/
COPY evaluation/ /app/evaluation/

ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

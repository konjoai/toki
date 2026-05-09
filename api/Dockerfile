# Toki API — small, slim, and self-contained.
# Build context must be the repo root so the COPY of python/ and api/ works.
#   docker build -f api/Dockerfile -t toki-api .
#   docker run -p 8000:8000 toki-api
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install API dependencies first (better layer caching).
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Install toki itself from source. We do not pull the [hf] extras — the API
# layer never needs torch/transformers; it only uses the pure-stdlib core.
COPY python/ /app/python/
RUN pip install --no-cache-dir /app/python

# API source last so code edits don't bust the dependency layer.
COPY api/ /app/api/

# Render and Fly.io both inject $PORT at runtime; default to 8000 locally.
ENV PORT=8000
EXPOSE 8000

# Single worker is the right default — toki's evaluator is CPU-bound stdlib
# code, so over-subscribing workers wastes RAM and adds queueing latency.
# Bump TOKI_API_WORKERS at deploy time if your instance has spare cores.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${TOKI_API_WORKERS:-1}"]

# Multi-stage Dockerfile for Idea Evolution
# Optimized for Cloud Run deployment

# Build stage
FROM ghcr.io/astral-sh/uv:0.11.16 AS uv
FROM python:3.11-slim-bookworm AS python-base
FROM python-base AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Resolve exactly the same versions as development, excluding test tools.
COPY --from=uv /uv /bin/uv
COPY pyproject.toml uv.lock ./
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN uv sync --locked --no-dev --no-install-project

# Runtime stage
FROM python-base AS runtime

WORKDIR /app

# Create non-root user
RUN useradd --create-home appuser

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Copy application code
COPY idea/ ./idea/

# Create data directory (for local development only)
RUN mkdir -p data && chown appuser:appuser data

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Cloud Run uses PORT environment variable
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8080') + '/', timeout=2)" || exit 1

# Run with production settings
CMD ["sh", "-c", "exec uvicorn idea.viewer:app --host 0.0.0.0 --port \"${PORT:-8080}\""]

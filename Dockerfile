FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.33 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY src ./src

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN uv sync --frozen --no-dev --no-editable

ENV COVE_DATA_DIR=/data \
    COVE_GEMINI_MODEL=gemini-3.7-flash \
    PATH=/app/.venv/bin:$PATH \
    PORT=8000 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["cove-sensory-mcp", "serve-http"]

FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS production
WORKDIR /app

RUN useradd --create-home --no-log-init appuser
USER appuser

COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser . .

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8001

# python:3.11-slim has no curl, so the health check uses Python's standard library.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health', timeout=5)" || exit 1

EXPOSE $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2"]

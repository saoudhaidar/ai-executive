# Dependency-free runtime: the standard library is the only requirement.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ASSISTANT_HOST=0.0.0.0 \
    ASSISTANT_PORT=8000 \
    ASSISTANT_DATABASE_PATH=/data/assistant.sqlite3 \
    ASSISTANT_PID_PATH=/data/assistant.pid

RUN useradd --create-home --uid 10001 assistant \
    && mkdir -p /data && chown assistant:assistant /data

WORKDIR /app
COPY --chown=assistant:assistant governed_assistant ./governed_assistant
COPY --chown=assistant:assistant web ./web
COPY --chown=assistant:assistant data ./data

USER assistant
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"

CMD ["python", "-m", "governed_assistant"]

# One image for every Python component: the four microservices, the load
# generator and the IncidentDNA backend. Which one runs is decided by the
# command in docker-compose.yml, so there is a single dependency set and a
# single build to keep in sync.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-services.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt -r requirements-services.txt

COPY incidentdna/ ./incidentdna/
COPY services/ ./services/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=4s --start-period=20s --retries=4 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

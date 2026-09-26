# IncidentDNA
#
#   make setup      create .venv and install dependencies
#   make demo       run the dashboard on the simulated system  (no Docker)
#   make pipeline   dataset -> train -> evaluate               (~4 minutes)
#   make up         run the real stack on Docker Compose
#
# The two paths meet at the telemetry bus: `make demo` drives the pipeline from
# the simulator, `make up` drives it from four real microservices. Everything
# downstream of the bus is the same code.

VENV    ?= .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
PORT    ?= 8000
SPEED   ?= 10

.DEFAULT_GOAL := help
.PHONY: help setup demo serve dataset train evaluate pipeline test lint \
        up down logs rebuild capture replay clean distclean results

help:                     ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- local dev
$(VENV)/bin/python:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

setup: $(VENV)/bin/python  ## create the virtualenv and install dependencies
	@echo "ready: use 'make demo' or 'make pipeline'"

setup-services: setup      ## also install the deps the real services need
	$(PIP) install -r requirements-services.txt

demo: setup                ## run the dashboard against the simulated system
	@echo "IncidentDNA -> http://localhost:$(PORT)   (simulating at $(SPEED)x)"
	INCIDENTDNA_SPEED=$(SPEED) $(PY) -m uvicorn backend.app:app --port $(PORT) --host 0.0.0.0

serve: demo                ## alias for demo

# ------------------------------------------------------------------ the ML
dataset: setup             ## generate the labelled telemetry dataset (~20s)
	$(PY) scripts/generate_dataset.py

train: setup               ## fit the detectors and the root-cause rankers (~70s)
	$(PY) scripts/train.py

evaluate: setup            ## score detection, diagnosis and throughput (~70s)
	$(PY) scripts/evaluate.py

pipeline: dataset train evaluate  ## the whole measured pipeline, end to end
	@echo
	@echo "results: experiments/results/report.md"

results:                   ## print the last evaluation report
	@cat experiments/results/report.md

# ------------------------------------------------------------------- tests
test: setup                ## run the test suite
	$(PYTEST) tests/ -q

test-verbose: setup        ## run the test suite with names
	$(PYTEST) tests/ -v

lint: setup                ## byte-compile everything as a cheap syntax check
	$(PY) -m compileall -q incidentdna backend services scripts tests

# ------------------------------------------------------------------ docker
up:                        ## build and run the real stack (services + Kafka + Postgres)
	docker compose up --build -d
	@echo "IncidentDNA  -> http://localhost:8000"
	@echo "API gateway  -> http://localhost:8080"
	@echo "inject a real fault:"
	@echo "  curl -XPOST localhost:8081/fault/database_latency -H 'content-type: application/json' -d '{\"severity\":0.85}'"

up-observability:          ## also run the OTel collector, Prometheus and Jaeger
	docker compose --profile observability up --build -d

down:                      ## stop the stack
	docker compose down -v

logs:                      ## follow the backend logs
	docker compose logs -f incidentdna

rebuild:                   ## rebuild images from scratch
	docker compose build --no-cache

# ------------------------------------------------------------ capture/replay
capture: setup             ## record the live demo's telemetry to a file
	INCIDENTDNA_CAPTURE=experiments/captures/demo.jsonl $(PY) -m uvicorn backend.app:app --port $(PORT)

replay: setup              ## replay a capture through the full pipeline
	$(PY) scripts/replay.py $(or $(FILE),experiments/captures/real_services_database_latency.jsonl.gz)

# ------------------------------------------------------------------- hygiene
clean:                     ## remove generated experiment outputs
	rm -rf experiments/dataset experiments/results experiments/artifacts
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache

distclean: clean           ## also remove the virtualenv
	rm -rf $(VENV)

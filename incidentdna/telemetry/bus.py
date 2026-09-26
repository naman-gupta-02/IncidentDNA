"""Telemetry transport.

The bus is the seam in the architecture. Producers (simulator or real
services) publish `TelemetryBatch`es; the feature pipeline subscribes. Three
implementations share one interface:

* `InMemoryBus`  - the local demo and the offline dataset builder.
* `FileBus`      - append-only JSONL, for capturing a run and replaying it.
* `KafkaBus`     - the Docker path, topics `telemetry.spans` / `.metrics` /
                   `.logs` / `.deployments`.

Only the Kafka implementation needs a broker, and it imports its client
lazily so the rest of the system runs with no Kafka installed at all.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Iterable, List, Optional

from .schema import DeploymentEvent, LogRecord, MetricPoint, Span, TelemetryBatch

TOPIC_SPANS = "telemetry.spans"
TOPIC_METRICS = "telemetry.metrics"
TOPIC_LOGS = "telemetry.logs"
TOPIC_DEPLOYMENTS = "telemetry.deployments"

_RECORD_TYPES = {
    TOPIC_SPANS: Span,
    TOPIC_METRICS: MetricPoint,
    TOPIC_LOGS: LogRecord,
    TOPIC_DEPLOYMENTS: DeploymentEvent,
}


class TelemetryBus:
    """Interface. `publish` is called by producers, `drain` by the pipeline."""

    def publish(self, batch: TelemetryBatch) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def drain(self) -> TelemetryBatch:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class InMemoryBus(TelemetryBus):
    """Thread-safe queue. `drain()` returns everything published since the
    last call, which maps exactly onto one window tick of the pipeline."""

    def __init__(self, max_backlog: int = 2_000_000) -> None:
        self._lock = threading.Lock()
        self._pending = TelemetryBatch()
        self._max_backlog = max_backlog
        self.published_records = 0
        self.dropped_records = 0

    def publish(self, batch: TelemetryBatch) -> None:
        with self._lock:
            if len(self._pending) > self._max_backlog:
                self.dropped_records += len(batch)
                return
            self._pending.extend(batch)
            self.published_records += len(batch)

    def drain(self) -> TelemetryBatch:
        with self._lock:
            out, self._pending = self._pending, TelemetryBatch()
        return out


class FileBus(TelemetryBus):
    """Append-only JSONL capture. Useful for `make capture` / replay and for
    shipping a reproducible telemetry sample in the repo."""

    def __init__(self, path: str | Path, mode: str = "a") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open(mode, encoding="utf-8")
        self._lock = threading.Lock()

    def publish(self, batch: TelemetryBatch) -> None:
        lines = []
        for topic, records in (
            (TOPIC_SPANS, batch.spans),
            (TOPIC_METRICS, batch.metrics),
            (TOPIC_LOGS, batch.logs),
            (TOPIC_DEPLOYMENTS, batch.deployments),
        ):
            for r in records:
                lines.append(json.dumps({"topic": topic, "record": r.to_dict()}))
        if not lines:
            return
        with self._lock:
            self._fh.write("\n".join(lines) + "\n")
            self._fh.flush()

    def drain(self) -> TelemetryBatch:
        return TelemetryBatch()

    def close(self) -> None:
        self._fh.close()

    @staticmethod
    def read(path: str | Path) -> TelemetryBatch:
        """Load a capture. `.gz` is handled transparently, because captures of
        a real run are large and are stored compressed."""
        import gzip

        path = Path(path)
        batch = TelemetryBatch()
        target = {
            TOPIC_SPANS: batch.spans,
            TOPIC_METRICS: batch.metrics,
            TOPIC_LOGS: batch.logs,
            TOPIC_DEPLOYMENTS: batch.deployments,
        }
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                cls = _RECORD_TYPES[payload["topic"]]
                target[payload["topic"]].append(cls(**payload["record"]))
        return batch


class TeeBus(TelemetryBus):
    """Publish to several buses at once (e.g. in-memory + capture file)."""

    def __init__(self, primary: TelemetryBus, *others: TelemetryBus) -> None:
        self.primary = primary
        self.others = list(others)

    def publish(self, batch: TelemetryBatch) -> None:
        self.primary.publish(batch)
        for bus in self.others:
            bus.publish(batch)

    def drain(self) -> TelemetryBatch:
        return self.primary.drain()

    def close(self) -> None:
        self.primary.close()
        for bus in self.others:
            bus.close()


class KafkaBus(TelemetryBus):
    """Kafka / Redpanda transport used by the Docker stack.

    Requires `kafka-python`. Producers use one topic per signal type so a
    consumer can subscribe to metrics only (cheap) or spans too (expensive).
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "incidentdna-pipeline",
        consume: bool = True,
        produce: bool = True,
        poll_timeout_ms: int = 250,
    ) -> None:
        try:
            from kafka import KafkaConsumer, KafkaProducer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "KafkaBus needs `pip install kafka-python`. The local demo "
                "path uses InMemoryBus and does not require Kafka."
            ) from exc

        self.poll_timeout_ms = poll_timeout_ms
        self._producer = None
        self._consumer = None
        if produce:
            self._producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode(),
                linger_ms=50,
                acks=1,
            )
        if consume:
            self._consumer = KafkaConsumer(
                TOPIC_SPANS,
                TOPIC_METRICS,
                TOPIC_LOGS,
                TOPIC_DEPLOYMENTS,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda v: json.loads(v.decode()),
                auto_offset_reset="latest",
                enable_auto_commit=True,
            )

    def publish(self, batch: TelemetryBatch) -> None:
        if self._producer is None:
            return
        for topic, records in (
            (TOPIC_SPANS, batch.spans),
            (TOPIC_METRICS, batch.metrics),
            (TOPIC_LOGS, batch.logs),
            (TOPIC_DEPLOYMENTS, batch.deployments),
        ):
            for r in records:
                self._producer.send(topic, r.to_dict())

    def drain(self) -> TelemetryBatch:
        batch = TelemetryBatch()
        if self._consumer is None:
            return batch
        target = {
            TOPIC_SPANS: batch.spans,
            TOPIC_METRICS: batch.metrics,
            TOPIC_LOGS: batch.logs,
            TOPIC_DEPLOYMENTS: batch.deployments,
        }
        polled = self._consumer.poll(timeout_ms=self.poll_timeout_ms, max_records=20000)
        for tp, messages in polled.items():
            cls = _RECORD_TYPES[tp.topic]
            sink = target[tp.topic]
            for msg in messages:
                sink.append(cls(**msg.value))
        return batch

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush()
            self._producer.close()
        if self._consumer is not None:
            self._consumer.close()


def make_bus(kind: str = "memory", **kwargs) -> TelemetryBus:
    kind = kind.lower()
    if kind == "memory":
        return InMemoryBus(**kwargs)
    if kind == "file":
        return FileBus(**kwargs)
    if kind == "kafka":
        return KafkaBus(**kwargs)
    raise ValueError(f"unknown bus kind: {kind}")

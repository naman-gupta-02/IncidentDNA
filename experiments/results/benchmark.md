# IncidentDNA benchmark

`Apple M1` · 8 logical cores · Python 3.11.1 · NumPy 2.4.6

Single process, single thread. Every stage is timed in isolation with warm-up and repeats; the throughput column is derived from the median and the tail columns are what determine whether a backlog forms.

## Pipeline stages

| stage | throughput | p50 | p95 | p99 |
|---|---|---|---|---|
| Telemetry ingest (bucketing records into windows) | 1,935,098 telemetry records/s | 0.336 ms | 0.346 ms | 0.369 ms |
| Feature reduction (buckets to feature rows) | 5,503 service-windows/s | 0.909 ms | 0.944 ms | 0.986 ms |
| Baseline resolution (median/MAD per metric) | 2,226 service-windows/s | 2.246 ms | 2.345 ms | 2.417 ms |
| Anomaly detection (4 detectors, 5 services) | 567 service-windows/s | 8.817 ms | 9.526 ms | 9.941 ms |
| Root-cause ranking + evidence (per incident) | 265 incidents/s | 3.780 ms | 4.103 ms | 4.202 ms |
|   candidate feature extraction only | 765 incidents/s | 1.307 ms | 1.335 ms | 1.529 ms |
| End to end (telemetry in, diagnosis out, one window) | 83 windows/s | 12.003 ms | 12.596 ms | 13.395 ms |

## Cost per detector

| detector | throughput | p50 | p99 |
|---|---|---|---|
| `static_threshold` | 745,379 service-windows/s | 0.007 ms | 0.014 ms |
| `rolling_zscore` | 152,476 service-windows/s | 0.033 ms | 0.035 ms |
| `isolation_forest` | 824 service-windows/s | 6.066 ms | 6.771 ms |
| `autoencoder` | 23,610 service-windows/s | 0.212 ms | 0.299 ms |

## Detector configuration: what the learned models cost

| configuration | throughput | p50 | p99 |
|---|---|---|---|
| `statistical_load_adjusted` | 2,164 service-windows/s | 2.311 ms | 2.674 ms |
| `full_ensemble` | 567 service-windows/s | 8.817 ms | 9.941 ms |

Dropping the Isolation Forest and the autoencoder makes detection **3.82x cheaper per window** — and the evaluation shows it is also the configuration with the better incident precision.

## Scaling with fleet size

| services | p50 per window | p99 per window | per service | of the 15s budget |
|---|---|---|---|---|
| 5 | 8.74 ms | 9.85 ms | 1748.1 µs | 0.066% |
| 10 | 11.12 ms | 12.50 ms | 1112.3 µs | 0.083% |
| 25 | 18.52 ms | 19.51 ms | 740.8 µs | 0.130% |
| 50 | 30.33 ms | 37.85 ms | 606.6 µs | 0.252% |
| 100 | 54.44 ms | 86.03 ms | 544.4 µs | 0.574% |

**Capacity:** order of magnitude **27,554 services** per process per 15s window — linear extrapolation from the largest measured fleet (100 services, 54.4 ms p50 per window) against one 15s window; an order-of-magnitude figure, not a measurement. Memory and Kafka consumer throughput would bind long before CPU does.

Peak RSS during the benchmark: 238.6 MB.

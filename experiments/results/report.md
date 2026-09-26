# IncidentDNA evaluation

Split: **test** — 45 runs, 36 injected incidents, 15s windows. Runs in this split were never used to fit a detector or a ranker.

## Detection

Alerts are matched to injected incidents as whole events, not rows.

| detector | precision | recall | F1 | median delay | p95 delay | false alerts/h |
|---|---|---|---|---|---|---|
| `zscore_only` | 0.705 | 0.861 | 0.775 | 30 s | 60 s | 1.11 |
| `statistical` | 0.821 | 0.889 | 0.853 | 30 s | 67 s | 0.60 |
| `statistical_load_adjusted` | 0.971 | 0.944 | 0.958 | 30 s | 65 s | 0.09 |
| `full_ensemble` | 0.944 | 0.944 | 0.944 | 30 s | 65 s | 0.17 |

### Recall by fault type (full ensemble)

| fault | n | recall | median delay |
|---|---|---|---|
| `bad_deployment` | 6 | 1.000 | 30 s |
| `connection_exhaustion` | 6 | 1.000 | 30 s |
| `cpu_saturation` | 6 | 1.000 | 30 s |
| `database_latency` | 6 | 0.833 | 30 s |
| `memory_leak` | 6 | 0.833 | 60 s |
| `payment_timeout` | 6 | 1.000 | 30 s |

## Diagnosis

Scored on the 34 incidents the full ensemble detected, with a mean of 4.5 candidate services per incident. All rankers see identical candidates.

| ranker | top-1 | top-3 | MRR | coverage |
|---|---|---|---|---|
| `random` | 0.176 | 0.676 | 0.471 | 1.000 |
| `naive_anomaly_strength` | 0.441 | 1.000 | 0.632 | 1.000 |
| `naive_earliest_onset` | 1.000 | 1.000 | 1.000 | 1.000 |
| `transparent` | 1.000 | 1.000 | 1.000 | 1.000 |
| `causal_locality` | 0.971 | 1.000 | 0.985 | 1.000 |
| `learned_logistic` | 1.000 | 1.000 | 1.000 | 1.000 |
| `learned_gbm` | 0.971 | 1.000 | 0.985 | 1.000 |

*Coverage* is the fraction of incidents where the true root cause was abnormal enough to become a candidate at all. It is the ceiling on top-1 for every ranker.

### Top-1 by fault type (`causal_locality`)

| fault | n | top-1 | top-3 | MRR |
|---|---|---|---|---|
| `bad_deployment` | 6 | 0.833 | 1.000 | 0.917 |
| `connection_exhaustion` | 6 | 1.000 | 1.000 | 1.000 |
| `cpu_saturation` | 6 | 1.000 | 1.000 | 1.000 |
| `database_latency` | 5 | 1.000 | 1.000 | 1.000 |
| `memory_leak` | 5 | 1.000 | 1.000 | 1.000 |
| `payment_timeout` | 6 | 1.000 | 1.000 | 1.000 |

### Where it goes wrong

| run | fault | true cause | predicted | rank of truth | mode |
|---|---|---|---|---|---|
| `run_0151` | `bad_deployment` | `api-gateway` | `postgres` | 2 | misrank |

## System performance

| metric | value |
|---|---|
| telemetry events/s | 110,427 |
| service-windows/s | 876 |
| mean window → diagnosis | 5.7 ms |
| p95 window → diagnosis | 7.9 ms |
| real-time speed-up | 2,629x |

Measured single-process on the replay path, which runs the same feature, detection and ranking code as the live system.

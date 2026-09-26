"""The evaluation maths itself — a metric bug would invalidate every result."""
import math

import pytest

from incidentdna.eval.detection import AlertRecord, TruthRecord, evaluate_detection
from incidentdna.eval.diagnosis import DiagnosisResult, RankedIncident

W = 15


def _alert(run, window, incident_id="i1"):
    return AlertRecord(run, incident_id, window, window, window + 10, 0.5)


def _truth(run, start=100, end=140, fault="database_latency"):
    return TruthRecord(run, fault, "postgres", start, end)


def test_an_alert_inside_the_incident_is_a_true_positive():
    res = evaluate_detection({"r1": [_alert("r1", 104)]}, {"r1": _truth("r1")}, {"r1": 96},
                             window_seconds=W)
    assert res.true_positives == 1 and not res.false_alerts and not res.missed
    assert res.precision == 1.0 and res.recall == 1.0


def test_an_alert_in_a_clean_run_is_a_false_alert():
    res = evaluate_detection({"r1": [_alert("r1", 50)]}, {"r1": None}, {"r1": 96}, window_seconds=W)
    assert res.true_positives == 0 and len(res.false_alerts) == 1
    assert res.precision == 0.0


def test_an_undetected_incident_is_a_miss_not_a_false_alert():
    res = evaluate_detection({"r1": []}, {"r1": _truth("r1")}, {"r1": 96}, window_seconds=W)
    assert len(res.missed) == 1 and not res.false_alerts
    assert res.recall == 0.0


def test_detection_delay_is_measured_from_the_injected_start():
    res = evaluate_detection({"r1": [_alert("r1", 106)]}, {"r1": _truth("r1", start=100)},
                             {"r1": 96}, window_seconds=W)
    assert res.matched[0]["delay_seconds"] == 6 * W


def test_a_second_concurrent_alert_is_a_duplicate_not_a_new_detection():
    alerts = [_alert("r1", 104, "i1"), _alert("r1", 112, "i2")]
    res = evaluate_detection({"r1": alerts}, {"r1": _truth("r1")}, {"r1": 96}, window_seconds=W)
    assert res.true_positives == 1 and res.duplicate_alerts == 1
    assert not res.false_alerts, "on-call sees one page, not two incidents"


def test_an_alert_after_the_grace_period_is_a_false_alert():
    res = evaluate_detection({"r1": [_alert("r1", 160)]}, {"r1": _truth("r1", end=140)},
                             {"r1": 96}, grace_windows=8, window_seconds=W)
    assert len(res.false_alerts) == 1 and len(res.missed) == 1


def test_false_alert_rate_is_per_hour_of_observed_normal_time():
    # 240 windows of 15s = 1 hour of clean observation, with two alerts.
    res = evaluate_detection(
        {"r1": [_alert("r1", 10, "a"), _alert("r1", 20, "b")]},
        {"r1": None}, {"r1": 240}, window_seconds=W,
    )
    assert res.normal_hours == pytest.approx(1.0)
    assert res.false_alerts_per_hour == pytest.approx(2.0)


def test_f1_combines_precision_and_recall():
    res = evaluate_detection(
        {"a": [_alert("a", 104)], "b": [], "c": [_alert("c", 10)]},
        {"a": _truth("a"), "b": _truth("b"), "c": None},
        {"a": 96, "b": 96, "c": 96}, window_seconds=W,
    )
    assert res.precision == pytest.approx(0.5)
    assert res.recall == pytest.approx(0.5)
    assert res.f1 == pytest.approx(0.5)


# --- diagnosis metrics ----------------------------------------------------
def _ranked(ranking, truth="postgres"):
    return RankedIncident("r", "i", "database_latency", truth, ranking)


def test_top1_top3_and_mrr():
    res = DiagnosisResult("x", [
        _ranked(["postgres", "order-service"]),                       # rank 1
        _ranked(["order-service", "postgres"]),                       # rank 2
        _ranked(["a", "b", "postgres"]),                              # rank 3
        _ranked(["a", "b", "c", "postgres"]),                         # rank 4
    ])
    assert res.top1 == pytest.approx(0.25)
    assert res.top3 == pytest.approx(0.75)
    assert res.mrr == pytest.approx((1 + 0.5 + 1 / 3 + 0.25) / 4)


def test_a_truth_outside_the_candidate_set_scores_zero_not_an_error():
    res = DiagnosisResult("x", [_ranked(["order-service", "api-gateway"])])
    assert res.incidents[0].rank is None
    assert res.mrr == 0.0 and res.top1 == 0.0
    assert res.candidate_coverage == 0.0


def test_coverage_is_the_ceiling_on_top1():
    res = DiagnosisResult("x", [
        _ranked(["postgres"]),
        _ranked(["order-service"]),
    ])
    assert res.candidate_coverage == 0.5
    assert res.top1 <= res.candidate_coverage


def test_failures_distinguish_a_miss_from_a_misrank():
    res = DiagnosisResult("x", [
        _ranked(["order-service", "postgres"]),   # misrank
        _ranked(["order-service"]),               # miss
    ])
    modes = {f["mode"] for f in res.failures()}
    assert modes == {"misrank", "miss"}

import importlib

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from service import monitoring
from conftest import REAL_FEATURES, REAL_MODEL, REAL_PREDS, REAL_REFERENCE, records_from_frame


# ---------------------------------------------------------------- PSI arithmetic
def test_psi_identical_distributions_is_zero():
    assert monitoring.psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)


def test_psi_matches_hand_calculation():
    e, a = np.array([0.5, 0.5]), np.array([0.25, 0.75])
    expected = (0.25 - 0.5) * np.log(0.25 / 0.5) + (0.75 - 0.5) * np.log(0.75 / 0.5)
    assert monitoring.psi(e, a) == pytest.approx(expected)


def test_psi_floors_empty_bins():
    assert np.isfinite(monitoring.psi([0.5, 0.5, 0.0], [0.0, 0.5, 0.5]))


def test_numeric_reference_counts_missing_separately():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
    ref = monitoring.numeric_reference(x, n_bins=2)
    shares = monitoring.current_shares(x, ref)
    assert shares[-1] == pytest.approx(0.2) and shares.sum() == pytest.approx(1.0)


def test_categorical_reference_puts_new_categories_in_other_bin():
    ref = monitoring.categorical_reference(pd.Series([0, 0, 1, 1]))
    shares = monitoring.current_shares(pd.Series([0, 1, 7, 7]), ref)
    assert shares.tolist() == [0.25, 0.25, 0.5]


def test_categorical_psi_of_a_distribution_with_missing_values_against_itself_is_zero():
    x = pd.Series([1, 1, 2, 2, 3, None, None, 1], dtype="Int64")
    ref = monitoring.categorical_reference(x)
    assert sum(ref["shares"]) == pytest.approx(1.0)
    assert monitoring.psi(ref["shares"], monitoring.current_shares(x, ref)) == pytest.approx(0.0)


def test_numeric_psi_of_a_distribution_with_missing_values_against_itself_is_zero():
    x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0])
    ref = monitoring.numeric_reference(x, n_bins=4)
    assert monitoring.psi(ref["shares"], monitoring.current_shares(x, ref)) == pytest.approx(0.0)


def test_status_thresholds():
    assert [monitoring.status(v) for v in (0.05, 0.10, 0.30)] == ["stable", "watch", "investigate"]


# ---------------------------------------------------------------- API mechanics (fixture model)
def test_health(fixture_service):
    svc, _ = fixture_service
    body = TestClient(svc.app).get("/health").json()
    assert body["status"] == "ok" and body["features"] == 33 and body["trees"] == 20


def test_score_returns_one_prediction_per_record(fixture_service):
    svc, X = fixture_service
    recs = records_from_frame(X.head(5), ["CA"] * 5, ["2024-07"] * 5)
    body = TestClient(svc.app).post("/score", json={"records": recs}).json()
    assert len(body["predictions"]) == 5
    for p in body["predictions"]:
        assert p["price"] == pytest.approx(np.exp(p["log_price"]))


def test_score_matches_direct_model_prediction(fixture_service):
    svc, X = fixture_service
    recs = records_from_frame(X.head(20), ["TX"] * 20, ["2024-03"] * 20)
    got = [p["log_price"] for p in TestClient(svc.app).post("/score", json={"records": recs}).json()["predictions"]]
    Xd = X.head(20).copy(); Xd["state_code"] = 1; Xd["moy"] = 3
    assert np.allclose(got, svc.model().predict(Xd[svc.FEATURES]))


def test_unseen_state_uses_unseen_code(fixture_service):
    svc, X = fixture_service
    client = TestClient(svc.app)
    recs = records_from_frame(X.head(3), ["WI"] * 3, ["2024-07"] * 3)
    body = client.post("/score", json={"records": recs}).json()
    assert not any(p["state_seen_in_training"] for p in body["predictions"])
    Xd = X.head(3).copy(); Xd["state_code"] = -1; Xd["moy"] = 7
    assert np.allclose([p["log_price"] for p in body["predictions"]], svc.model().predict(Xd[svc.FEATURES]))


def test_missing_features_are_allowed(fixture_service):
    svc, X = fixture_service
    rec = records_from_frame(X.head(1), ["CA"], ["2024-07"])[0]
    rec["comp_med20"] = None
    assert TestClient(svc.app).post("/score", json={"records": [rec]}).status_code == 200


@pytest.mark.parametrize("change", [{"sale_month": "2024-13"}, {"state": "california"}, {"unknown_field": 1.0},
                                    {"log_sqft": "big"}])
def test_invalid_records_are_rejected(fixture_service, change):
    svc, X = fixture_service
    rec = records_from_frame(X.head(1), ["CA"], ["2024-07"])[0]
    rec.update(change)
    assert TestClient(svc.app).post("/score", json={"records": [rec]}).status_code == 422


def test_empty_batch_is_rejected(fixture_service):
    svc, _ = fixture_service
    assert TestClient(svc.app).post("/score", json={"records": []}).status_code == 422


def test_psi_endpoint_needs_enough_records(fixture_service):
    svc, X = fixture_service
    recs = records_from_frame(X.head(50), ["CA"] * 50, ["2024-07"] * 50)
    assert TestClient(svc.app).post("/monitor/psi", json={"records": recs}).status_code == 422


def test_psi_endpoint_stable_on_reference_like_data_and_flags_shift(fixture_service):
    svc, X = fixture_service
    client = TestClient(svc.app)
    n = 1000
    rng = np.random.default_rng(5)
    months = [f"2024-{m:02d}" for m in rng.integers(1, 13, n)]
    states = list(np.where(rng.integers(0, 2, n) == 0, "CA", "TX"))
    stable = client.post("/monitor/psi", json={"records": records_from_frame(X.sample(n, random_state=3), states, months)}).json()
    assert stable["investigate"] == [] and stable["prediction_psi"] < 0.10
    shifted = X.sample(n, random_state=4).copy(); shifted["log_sqft"] += 2.0
    body = client.post("/monitor/psi", json={"records": records_from_frame(shifted, states, months)}).json()
    assert "log_sqft" in body["investigate"]
    single_month = client.post("/monitor/psi", json={"records": records_from_frame(X.sample(n, random_state=3), states, ["2024-07"] * n)}).json()
    assert "moy" not in single_month["investigate"] and single_month["expected_shift"]["moy"] >= 0.25


# ---------------------------------------------------------------- parity with the real model (local only)
@pytest.mark.skipif(not (REAL_MODEL.exists() and REAL_FEATURES.exists() and REAL_PREDS.exists() and REAL_REFERENCE.exists()),
                    reason="real model and de-identified data are only available locally")
def test_real_model_reproduces_offline_predictions(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(REAL_MODEL))
    monkeypatch.setenv("REFERENCE_PATH", str(REAL_REFERENCE))
    from service import app as svc
    svc = importlib.reload(svc)
    S = pd.read_parquet(REAL_FEATURES)
    P = pd.read_parquet(REAL_PREDS, columns=["addr_id", "c1_gbm"])
    idx = np.concatenate([np.flatnonzero(S.split.eq(s).to_numpy())[:250] for s in ("oot", "spatial")])
    sub = S.iloc[idx]
    months = ((sub.t_sale // 12 + 1990).astype(str) + "-" + (sub.t_sale % 12 + 1).astype(str).str.zfill(2)).tolist()
    recs = records_from_frame(sub, sub.state.tolist(), months)
    body = TestClient(svc.app).post("/score", json={"records": recs}).json()
    got = np.array([p["log_price"] for p in body["predictions"]])
    assert np.abs(got - P.c1_gbm.to_numpy()[idx]).max() < 1e-9
    svc.model.cache_clear(); svc.model_sha.cache_clear(); svc.reference.cache_clear()

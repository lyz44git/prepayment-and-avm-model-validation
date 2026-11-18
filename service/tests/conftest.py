"""Test fixtures.

`fixture_service` points the app at a tiny LightGBM model and reference trained on random numbers, so the
API mechanics can be tested anywhere (including CI, where the real model and data are not available).
The fixture model is for testing only; no reported result uses it.
"""
import importlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from service import monitoring  # noqa: E402

REAL_MODEL = ROOT / "data_work" / "p2" / "c1_gbm.txt"
REAL_FEATURES = ROOT / "data_work" / "p2" / "features.parquet"
REAL_PREDS = ROOT / "data_work" / "p2" / "preds_gbm.parquet"
REAL_REFERENCE = ROOT / "service" / "reference" / "psi_reference.json"


def _fixture_frame(features, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(features))), columns=features)
    X["state_code"] = rng.integers(0, 2, n)
    X["moy"] = rng.integers(1, 13, n)
    X["pool"] = rng.integers(0, 2, n)
    X["condo"] = rng.integers(0, 2, n)
    return X


@pytest.fixture()
def fixture_service(tmp_path, monkeypatch):
    from service import app as svc

    X = _fixture_frame(svc.FEATURES)
    y = 12 + 0.3 * X["log_sqft"] + 0.2 * X["comp_med20"] + np.random.default_rng(1).normal(0, 0.1, len(X))
    booster = lgb.train({"objective": "regression", "verbose": -1, "num_leaves": 7, "seed": 0},
                        lgb.Dataset(X, y, categorical_feature=["state_code", "moy"]), 20)
    model_path = tmp_path / "model.txt"
    booster.save_model(str(model_path))
    pred = booster.predict(X)
    ref = {"state_codes": {"CA": 0, "TX": 1, "WI": 2}, "training_states": ["CA", "TX"], "columns": {},
           "expected_shift": {"moy": "seasonal", "mkt_idx": "calendar time"}}
    for c in svc.FEATURES:
        ref["columns"][c] = (monitoring.categorical_reference(X[c]) if c in {"state_code", "moy", "pool", "condo"}
                             else monitoring.numeric_reference(X[c]))
    ref["columns"]["prediction"] = monitoring.numeric_reference(pd.Series(pred))
    ref_path = tmp_path / "reference.json"
    ref_path.write_text(json.dumps(ref))
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("REFERENCE_PATH", str(ref_path))
    svc = importlib.reload(svc)
    yield svc, X
    svc.model.cache_clear(); svc.model_sha.cache_clear(); svc.reference.cache_clear()


def records_from_frame(X, states, months):
    from service import app as svc
    out = []
    for i, row in enumerate(X[svc.INPUT_FEATURES].itertuples(index=False)):
        r = {k: (None if pd.isna(v) else float(v)) for k, v in zip(svc.INPUT_FEATURES, row)}
        r["state"] = states[i]
        r["sale_month"] = months[i]
        out.append(r)
    return out

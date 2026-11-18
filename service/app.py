"""FastAPI scoring service for the valuation study's LightGBM challenger (C1).

Run from the prepayment-and-avm-model-validation folder:
    MODEL_PATH=data_work/p2/c1_gbm.txt uvicorn service.app:app --port 8000

Endpoints
- GET  /health        model fingerprint, tree and feature counts
- POST /score         log price and price for 1-10,000 records
- POST /monitor/psi   PSI of the batch's features and predictions against the training-window reference

Features are expected to arrive precomputed (as from a feature store), with the same definitions as
scripts/11_p2_features.py. The service derives `state_code` and `moy` itself, and treats a state that was not
in the training data as unseen (code -1), exactly as the offline spatial evaluation did.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from .monitoring import load_reference, psi_table

HERE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", HERE.parent / "data_work" / "p2" / "c1_gbm.txt"))
REFERENCE_PATH = Path(os.environ.get("REFERENCE_PATH", HERE / "reference" / "psi_reference.json"))
MAX_RECORDS = 10_000
MIN_PSI_RECORDS = 200

# order must match scripts/13_p2_challenger.py
FEATURES = ["prev_idx_logp", "months_since_prev", "log_mv0", "log_av0", "log_cv0", "log_tax0", "age", "log_sqft", "beds",
            "baths", "log_lot", "pool", "condo", "stories0", "units0", "comp_med5", "comp_med20", "comp_iqr20",
            "comp_lppsf_med20", "comp_dist1", "comp_dist20", "comp_lsqft_med20", "comp_age_med20", "comp_n_1km",
            "comp_implied", "mkt_idx", "lat", "lon", "log_density", "t_owned", "t_medage", "state_code", "moy"]
DERIVED = {"state_code", "moy"}
INPUT_FEATURES = [f for f in FEATURES if f not in DERIVED]

Record = create_model(
    "Record",
    __config__=ConfigDict(extra="forbid"),
    state=(str, Field(pattern=r"^[A-Z]{2}$", description="two-letter state code")),
    sale_month=(str, Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$", description="YYYY-MM")),
    **{f: (Optional[float], None) for f in INPUT_FEATURES},
)


class Batch(BaseModel):
    records: list[Record] = Field(min_length=1, max_length=MAX_RECORDS)


class Scored(BaseModel):
    log_price: float
    price: float
    state_seen_in_training: bool


class ScoreResponse(BaseModel):
    model_sha256: str
    predictions: list[Scored]


@lru_cache(maxsize=1)
def model() -> lgb.Booster:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"model file not found: {MODEL_PATH}")
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    if booster.feature_name() != FEATURES:
        raise RuntimeError("model features do not match the service feature list")
    return booster


@lru_cache(maxsize=1)
def model_sha() -> str:
    return hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:16]


@lru_cache(maxsize=1)
def reference() -> dict:
    return load_reference(REFERENCE_PATH)


def to_frame(records: list) -> tuple[pd.DataFrame, np.ndarray]:
    """Model matrix in training column order, plus a flag for states seen in training."""
    raw = pd.DataFrame([r.model_dump() for r in records])
    ref = reference()
    codes = ref["state_codes"]                      # alphabetical codes over all states, as in training
    trained = set(ref["training_states"])
    seen = raw.state.isin(trained).to_numpy()
    X = raw[INPUT_FEATURES].astype(float)
    X["state_code"] = np.where(seen, raw.state.map(codes).fillna(-1), -1).astype(int)
    X["moy"] = raw.sale_month.str.slice(5, 7).astype(int)
    return X[FEATURES], seen


def predict(X: pd.DataFrame) -> np.ndarray:
    return model().predict(X)


app = FastAPI(title="Valuation challenger scoring service", version="1.0")


@app.get("/health")
def health() -> dict:
    m = model()
    return {"status": "ok", "model_sha256": model_sha(), "trees": m.num_trees(), "features": m.num_feature()}


@app.post("/score", response_model=ScoreResponse)
def score(batch: Batch) -> ScoreResponse:
    X, seen = to_frame(batch.records)
    yhat = predict(X)
    return ScoreResponse(model_sha256=model_sha(), predictions=[
        Scored(log_price=float(v), price=float(np.exp(v)), state_seen_in_training=bool(s)) for v, s in zip(yhat, seen)])


@app.post("/monitor/psi")
def monitor_psi(batch: Batch) -> dict:
    if len(batch.records) < MIN_PSI_RECORDS:
        raise HTTPException(status_code=422, detail=f"PSI needs at least {MIN_PSI_RECORDS} records")
    X, _ = to_frame(batch.records)
    frame = X.copy()
    frame["prediction"] = predict(X)
    table = psi_table(frame, reference())
    expected = reference().get("expected_shift", {})
    alert = table[~table.column.isin(expected)]
    return {"records": len(frame), "model_sha256": model_sha(),
            "prediction_psi": float(table.loc[table.column == "prediction", "psi"].iloc[0]),
            "investigate": alert[alert.status == "investigate"].column.tolist(),
            "watch": alert[alert.status == "watch"].column.tolist(),
            "expected_shift": {r.column: round(r.psi, 4) for r in table.itertuples() if r.column in expected},
            "note": "PSI covers inputs and predictions only; accuracy decay needs error monitoring on recorded sales.",
            "table": table.round(4).to_dict(orient="records")}

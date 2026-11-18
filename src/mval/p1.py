"""Prepayment study: property-month panel, features and cause-specific discrete-time hazard models."""
from __future__ import annotations
import numpy as np, pandas as pd
from .macro import monthly_rates

T0 = (2022 - 1990) * 12 + 4          # 2022-05 as month index (months since 1990-01)
T_END = (2025 - 1990) * 12 + 5       # 2025-06 (last month of the risk window)
TRAIN_END = (2024 - 1990) * 12 + 5   # 2024-06 (last training month; OOT test = 2024-07..2025-06)
HOLDOUT_STATES = ["MA", "NJ", "WI", "MN", "SC", "KY", "OK", "AZ", "WA"]   # one per census division, fixed a priori
MONTHS = pd.date_range("1990-01-01", "2027-12-01", freq="MS")

def rate_vector() -> np.ndarray:
    r = monthly_rates("1990-01-01").pmms30.reindex(MONTHS).ffill()
    return r.to_numpy()

def hpi_matrix(hpi: pd.DataFrame, lookup: pd.DataFrame, version: str = "full") -> tuple[dict, np.ndarray]:
    """Return {state: row} and an array [n_keys, len(MONTHS)] of log HPI (flat after last month)."""
    h = hpi[hpi.version == version].pivot(index="state", columns="month", values="log_hpi")
    h = h.reindex(columns=MONTHS).ffill(axis=1)
    key = lookup.set_index("state").hpi_key
    states = sorted(key.index)
    arr = np.vstack([h.loc[key[s]].to_numpy() for s in states])
    return {s: i for i, s in enumerate(states)}, arr

def build_panel(u: pd.DataFrame, pm: np.ndarray, st_idx: dict, loghpi: np.ndarray,
                cohort_rate: pd.Series | None = None) -> pd.DataFrame:
    """Expand properties to property-months [T0, min(event, T_END)] with time-varying covariates."""
    u = u.reset_index(drop=True)
    last = np.where(u.event.ne("none") & u.t_event.notna(), u.t_event.fillna(T_END), T_END).astype(int)
    last = np.minimum(last, T_END)
    n_m = (last - T0 + 1).clip(min=0)
    rows = np.repeat(np.arange(len(u)), n_m)
    t = T0 + (np.arange(n_m.sum()) - np.repeat(np.cumsum(n_m) - n_m, n_m))
    p = u.loc[rows].reset_index(drop=True)
    p["t"] = t
    is_last = t == last[rows]
    for c in ["sale", "new_lien", "other_transfer"]:
        p[f"y_{c}"] = (is_last & (u.event.values[rows] == c) & (u.t_event.values[rows] == t)).astype(np.int8)
    tp = p.t_purchase.to_numpy().astype(int)
    p["tenure_m"] = t - tp
    p["pmms"] = pm[t]
    p["note_hat"] = pm[tp]
    p["gap"] = p.pmms - p.note_hat                                   # lock-in: + means owner's rate below market
    if cohort_rate is not None:                                       # regression-calibration proxy (sensitivity)
        p["note_rc"] = cohort_rate.reindex(tp).to_numpy()
        p["gap_rc"] = p.pmms - p.note_rc
    p["pmms_chg3"] = p.pmms - pm[t - 3]
    si = p.state.map(st_idx).to_numpy()
    p["log_appr"] = loghpi[si, t] - loghpi[si, tp]                    # log price appreciation since purchase
    # estimated current LTV: 80% LTV at purchase, 30y level-pay at the purchase-month rate, value = price x HPI
    r = p.note_hat.to_numpy() / 1200; n = 360; k = np.minimum(p.tenure_m.to_numpy(), n)
    bal_frac = ((1 + r) ** n - (1 + r) ** k) / ((1 + r) ** n - 1)
    p["est_ltv"] = 0.8 * bal_frac / np.exp(p.log_appr.to_numpy())
    p["log_value"] = np.log(p.price0) + (loghpi[si, T0 - 1] - loghpi[si, tp])   # purchase price indexed to 2022-04
    p["moy"] = (t % 12) + 1
    return p

STATIC = ["log_value", "age", "log_sqft", "beds", "baths", "log_lot", "pool", "condo", "log_density", "t_owned", "t_medage"]
TV = ["tenure_m", "gap", "pmms", "pmms_chg3", "log_appr", "est_ltv"]

def add_static(u: pd.DataFrame) -> pd.DataFrame:
    u = u.copy()
    u["age"] = (2022 - u.yr_built).where(u.yr_built.between(1800, 2022))
    u["log_sqft"] = np.log(u.sqft.where(u.sqft.between(300, 20000)))
    u["log_lot"] = np.log1p(u.lot_sqft.where(u.lot_sqft.between(0, 5e7)))
    u["beds"] = u.beds.where(u.beds.between(0, 15)); u["baths"] = u.baths.where(u.baths.between(0, 15))
    u["condo"] = (u.proptype0 == "11").astype(int); u["pool"] = u.pool.astype(int)
    u["log_density"] = np.log1p(u.t_density)
    return u

# ---------------------------------------------------------------- design for the logit hazard
def ns_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis (truncated power form, linear beyond boundary knots)."""
    K = len(knots); x = np.asarray(x, float)
    def d(k):
        return (np.maximum(x - knots[k], 0) ** 3 - np.maximum(x - knots[-1], 0) ** 3) / (knots[-1] - knots[k])
    cols = [x] + [d(k) - d(K - 2) for k in range(K - 2)]
    return np.column_stack(cols)

class LogitDesign:
    """Fixed, documented design: tenure spline + rate gap (piecewise) + equity + controls + seasonality + state FE."""
    def __init__(self, gap_col: str = "gap", tenure_knots=(6, 36, 84, 150, 240, 360)):
        self.gap_col = gap_col; self.knots = np.array(tenure_knots, float)
    def fit(self, p: pd.DataFrame):
        self.med = p[STATIC].median(); self.states = sorted(p.state.unique())
        X = self._raw(p.sample(min(len(p), 400_000), random_state=0))
        X = X.loc[:, (X.std() > 0).to_numpy()]
        _, first = np.unique(X.to_numpy().T, axis=0, return_index=True)   # exact duplicate columns
        self.columns = [X.columns[i] for i in sorted(first)]; return self
    def transform(self, p: pd.DataFrame) -> pd.DataFrame:
        return self._raw(p).reindex(columns=self.columns, fill_value=0).astype(np.float32)
    def _raw(self, p: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=p.index)
        ten = np.clip(p.tenure_m.to_numpy(float), 1, 420)
        B = ns_basis(ten / 12, self.knots / 12)
        for j in range(B.shape[1]):
            X[f"tenure_ns{j}"] = B[:, j]
        g = p[self.gap_col].to_numpy(float)
        X["gap_pos"] = np.maximum(g, 0); X["gap_neg"] = np.minimum(g, 0)   # lock-in vs in-the-money
        X["pmms"] = p.pmms; X["pmms_chg3"] = p.pmms_chg3
        X["log_appr"] = p.log_appr.clip(-1.5, 2.5); X["est_ltv"] = p.est_ltv.clip(0, 2)
        for c in STATIC:
            v = p[c].astype(float); X[c] = v.fillna(self.med[c]); X[f"{c}_na"] = v.isna().astype(int)
        for m in range(2, 13):
            X[f"moy{m}"] = (p.moy == m).astype(int)
        for s in self.states[1:]:
            X[f"st_{s}"] = (p.state == s).astype(int)
        return X

LGB_FEATURES = TV + STATIC + ["moy", "state_code"]

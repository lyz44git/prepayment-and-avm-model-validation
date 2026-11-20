"""Small-business credit-loss study: constants, macro inputs and the logit design (plan: docs/credit_loss_plan.md)."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np, pandas as pd
from .p1 import ns_basis

def mi(year: int, month: int) -> int:
    """Month index: months since 2000-01 (2000-01 = 0)."""
    return (year - 2000) * 12 + (month - 1)

W0 = mi(2006, 1)            # first month of the risk window
W1 = mi(2026, 9)            # last month (data as of 2026-09-30)
FIT_END = mi(2017, 12)      # development fit months 2006-01 .. 2017-12
VAL_END = mi(2019, 12)      # validation 2018-01 .. 2019-12; refit through here for the out-of-time test
OOT0 = mi(2020, 1)
HOLDOUT_STATES = ["MA", "NJ", "WI", "MN", "SC", "KY", "OK", "AZ", "WA"]
SAMPLE_RATE = 0.05          # share of non-event loan-months kept
LOG_RATE = float(np.log(SAMPLE_RATE))
ORIGINS = [mi(y, 12) for y in (2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023)]
AGE_KNOTS = np.array([6, 12, 24, 48, 84, 120, 180], float)
STATES51 = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY",
            "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
            "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

def month_label(m: int) -> str:
    return f"{2000 + m // 12}-{m % 12 + 1:02d}"

def plan_sha(root: Path) -> str:
    return hashlib.sha256((root / "docs" / "credit_loss_plan.md").read_bytes()).hexdigest()

# ---------------------------------------------------------------- macro inputs (Fed 2026 files, quarterly -> monthly)
FED_COLS = {"Unemployment rate": "ur", "BBB corporate yield": "bbb", "10-year Treasury yield": "t10", "Prime rate": "prime"}

def _read_fed(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    q = d["Date"].str.extract(r"(\d{4}) Q(\d)").astype(int)
    d["year"], d["q"] = q[0], q[1]
    return d.rename(columns=FED_COLS)[["year", "q", *FED_COLS.values()]]

RELIEF_QUARTERS = [(2020, 2), (2020, 3), (2020, 4), (2021, 1), (2021, 2), (2021, 3)]   # SBA paid 7(a) payments

def macro_quarterly(fed_dir: Path, scenario: str = "baseline", relief_overlay: bool = False) -> pd.DataFrame:
    """History to 2025Q4 from the historic file, then the named supervisory scenario from 2026Q1 (to 2029Q1).
    relief_overlay (deviation D5, overlay O1): unemployment and the BBB spread for 2020Q2-2021Q3 are held at their
    2020Q1 values before the 4-quarter changes are computed."""
    hist = _read_fed(fed_dir / "2026_Final_Historic_Domestic.csv")
    name = {"baseline": "Baseline", "severe": "Severely_Adverse"}[scenario]
    scen = _read_fed(fed_dir / f"2026_Final_Supervisory_{name}_Domestic.csv")
    q = pd.concat([hist[hist.year <= 2025], scen[scen.year >= 2026]]).sort_values(["year", "q"]).reset_index(drop=True)
    if relief_overlay:
        base = q[(q.year == 2020) & (q.q == 1)].iloc[0]
        sel = pd.Series(list(zip(q.year, q.q))).isin(RELIEF_QUARTERS).to_numpy()
        for c in ("ur", "bbb", "t10"):
            q.loc[sel, c] = base[c]
    q["dur4"] = q.ur - q.ur.shift(4)
    q["bbb_spread"] = q.bbb - q.t10
    q["dprime4"] = q.prime - q.prime.shift(4)
    return q

CHANGES = {"dur4": "ur", "dprime4": "prime"}   # 4-quarter change -> its level


def macro_monthly(q: pd.DataFrame, last_month: int, revert_quarters: int = 4,
                  changes_from_levels: bool = True) -> pd.DataFrame:
    """Monthly table indexed by month index (from 1999-01), quarterly values applied to each month of the quarter.
    After the last scenario quarter, the levels (unemployment, BBB spread, prime) revert in a straight line over
    `revert_quarters` quarters to their 2006-2025 average and stay there; the 4-quarter changes are then the 12-month
    differences of the reverted levels, so levels and changes describe the same path (since
    2026-11-19). changes_from_levels=False reproduces the first published runs, which reverted each change
    to its own average independently of its level (reports/p3/before_2026-11-19/)."""
    rows = []
    for _, r in q.iterrows():
        for k in range(3):
            rows.append({"m": mi(int(r.year), int((r.q - 1) * 3 + 1 + k)), **{c: r[c] for c in ["ur", "dur4", "bbb_spread", "prime", "dprime4"]}})
    m = pd.DataFrame(rows).set_index("m")
    m = m[m.index >= mi(1999, 1)]
    hist = m[(m.index >= mi(2006, 1)) & (m.index <= mi(2025, 12))]
    lr = hist.mean()
    end = m.index.max()
    if last_month > end:
        last = m.loc[end]
        ext = []
        for t in range(end + 1, last_month + 1):
            w = min((t - end) / (3 * revert_quarters), 1.0)
            ext.append(pd.Series(last * (1 - w) + lr * w, name=t))
        m = pd.concat([m, pd.DataFrame(ext)])
        if changes_from_levels:
            for chg, lev in CHANGES.items():
                m.loc[m.index > end, chg] = (m[lev] - m[lev].shift(12))[m.index > end]
    m.attrs["long_run"] = lr.to_dict()
    return m


def start_survival(idx: np.ndarray, hco: np.ndarray, hpf: np.ndarray):
    """Probability that a loan is still open at the start of each projected month: S_0 = 1, S_t = S_{t-1}(1 - c_t - p_t).
    Rows are grouped by loan (idx) and ordered by month within each loan. Returns (S_{t-1}, rows where c_t + p_t > 1,
    the largest c_t + p_t). Replaces cumprod(q) / q, which returned 0 instead of S_{t-1} whenever q_t = 0."""
    tot = hco + hpf
    q = np.clip(1 - tot, 0, 1)
    first = np.r_[True, idx[1:] != idx[:-1]]
    qs = np.r_[1.0, q[:-1]]
    qs[first] = 1.0
    s_prev = pd.Series(qs).groupby(idx).cumprod().to_numpy()
    return s_prev, int((tot > 1).sum()), float(tot.max()) if len(tot) else 0.0

def downturn_exposure(dur4_by_month: pd.Series, t: np.ndarray, disb_m: np.ndarray, lag: int = 12, window: int = 36,
                      frozen_at: int | None = None) -> np.ndarray:
    """Largest 4-quarter rise in unemployment dated between max(first disbursement, t - lag - window) and t - lag,
    floored at 0 (deviation D5). With frozen_at, months after it take the value at frozen_at (macro held at origin)."""
    s = dur4_by_month
    t = np.asarray(t); d = np.asarray(disb_m)
    out = np.zeros(len(t))
    for k in range(window + 1):
        m = t - lag - k
        if frozen_at is not None:
            m = np.minimum(m, frozen_at)
        v = s.reindex(m).to_numpy()
        v = np.where((t - lag - k >= d) & ~np.isnan(v), v, 0.0)
        out = np.maximum(out, v)
    return np.maximum(out, 0.0)

# ---------------------------------------------------------------- logit design
SECTORS = ["11", "21", "22", "23", "31", "42", "44", "48", "51", "52", "53", "54", "56", "61", "62", "71", "72", "81"]

ERA_EDGES = [2005, 2009, 2014, 2020]          # approval-era effects (deviation D4): FY2000-04 | 05-08 | 09-13 | 14-19 | 20+

def era_of(approval_fy) -> np.ndarray:
    return np.searchsorted(ERA_EDGES, np.asarray(approval_fy), side="right")

class CreditDesign:
    """Design from the plan without the term-based features (deviation D1): age spline, loan and borrower attributes,
    month of year, state FE, and the macro terms of one cause."""
    def __init__(self, cause: str, lag: int = 0, eras: bool = False, downturn: bool = False):
        self.cause, self.lag, self.eras, self.downturn = cause, lag, eras, downturn
    def fit(self, p: pd.DataFrame):
        self.states = sorted(p.state.unique())
        e = era_of(p.approval_fy)                       # an era gets its own effect only with >= 50 events in the fit
        ev = pd.Series(e[p.y.to_numpy() == 1]).value_counts() if "y" in p.columns else pd.Series(1000, index=np.unique(e))
        self.max_era = int(max([k for k, n in ev.items() if n >= 50], default=0))
        X = self._raw(p.sample(min(len(p), 300_000), random_state=0))
        X = X.loc[:, (X.std() > 0).to_numpy()]
        self.columns = list(X.columns)
        return self
    def transform(self, p: pd.DataFrame) -> pd.DataFrame:
        return self._raw(p).reindex(columns=self.columns, fill_value=0).astype(np.float32)
    def macro_cols(self) -> list[str]:
        return (["dur4", "bbb_spread"] if self.cause == "co" else ["prime", "dprime4", "dur4"])
    def _raw(self, p: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=p.index)
        age = np.clip(p.age.to_numpy(float), 1, 400)
        B = ns_basis(age, AGE_KNOTS)
        for j in range(B.shape[1]):
            X[f"age_ns{j}"] = B[:, j]
        X["log_amount"] = np.log(p.gross_approval.clip(1000, 1e8))
        X["gty_share"] = p.gty_share.clip(0, 1)
        for g in ["express", "preferred", "lowdoc", "other"]:
            X[f"proc_{g}"] = (p.proc == g).astype(np.int8)
        for g in ["startup", "new", "unknown"]:
            X[f"bage_{g}"] = (p.bage == g).astype(np.int8)
        for g in ["individual", "partnership"]:
            X[f"btype_{g}"] = (p.btype == g).astype(np.int8)
        for s in SECTORS + ["na"]:
            X[f"naics_{s}"] = (p.naics2 == s).astype(np.int8)
        X["franchise"] = p.franchise.astype(np.int8)
        for k in range(2, 13):
            X[f"moy{k}"] = (p.moy == k).astype(np.int8)
        for c in self.macro_cols():
            X[c] = p[c].astype(float)
        if getattr(self, "downturn", False):            # deviation D5: default pipeline after a downturn
            X["dx"] = p.dx.astype(float)
        if self.eras:                                   # loans from an era not yet in the fit take the latest fitted era
            e = np.minimum(era_of(p.approval_fy), getattr(self, "max_era", 4))
            for k in range(1, 5):
                X[f"era{k}"] = (e == k).astype(np.int8)
        for s in self.states[1:]:
            X[f"st_{s}"] = (p.state == s).astype(np.int8)
        return X

LGB_CAT = ["proc", "bage", "btype", "naics2", "state"]
LGB_NUM = ["age", "log_amount", "gty_share", "franchise", "moy"]

def lgb_frame(p: pd.DataFrame, macro: list[str]) -> pd.DataFrame:
    X = pd.DataFrame({"age": p.age,
                      "log_amount": np.log(p.gross_approval.clip(1000, 1e8)), "gty_share": p.gty_share,
                      "franchise": p.franchise.astype(int), "moy": p.moy}, index=p.index)
    for c in LGB_CAT:
        X[c] = p[c].astype("category")
    for c in macro:
        X[c] = p[c].astype(float)
    return X

# ---------------------------------------------------------------- amortisation (balance as a share of the original amount)
def balance_share(rate_pct: np.ndarray, term: np.ndarray, age: np.ndarray) -> np.ndarray:
    r = np.asarray(rate_pct, float) / 1200; n = np.asarray(term, float); k = np.minimum(np.asarray(age, float), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = ((1 + r) ** n - (1 + r) ** k) / ((1 + r) ** n - 1)
        b = np.where(r > 0, b, 1 - k / n)
    return np.clip(b, 0, 1)

"""Fair-lending guardrails, reported even though the use case is marketing propensity.

* Adverse impact ratio (AIR): selection = ``score >= quantile(score, 0.9)``; ``AIR_g =
  sel_rate_g / sel_rate_ref`` with the largest group as reference; flagged when ``AIR < 0.8``
  (four-fifths rule).
* Standardised mean difference: ``(mean_g - mean_ref) / pooled_sd``.
* Proxy test: OLS ``score ~ C(ethnicity) + C(language) + C(gender) + tract shares``; the R² says
  how much of the score protected attributes can reconstruct, the coefficients where.
* Feature screen: Cramér's V (categorical pairs, numerics binned into deciles) or |Spearman|
  (numeric pairs) between every feature and every protected attribute; flag above 0.3.
* Less discriminatory alternative: drop flagged features, refit, and report the performance /
  fairness trade-off.

Protected attributes are read from the snapshot only inside this module and :mod:`altdata.audit`;
they never reach a feature matrix (:func:`altdata.features.assert_no_prohibited`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, spearmanr

from .metrics import auc

log = logging.getLogger(__name__)

AIR_FLAG_THRESHOLD = 0.8


def _groups(group) -> pd.Series:
    g = pd.Series(group).reset_index(drop=True)
    return g.astype("str").mask(g.isna())


def _reference(g: pd.Series, reference: str) -> str:
    return g.value_counts().idxmax() if reference == "largest" else reference


def adverse_impact_ratio(score, group, cutoff_quantile: float = 0.9, reference: str = "largest") -> pd.DataFrame:
    """AIR per group at the top-``(1 - cutoff_quantile)`` selection cut; rows with NA group are dropped."""
    s = pd.Series(np.asarray(score, dtype=float)).reset_index(drop=True)
    g = _groups(group)
    ok = s.notna() & g.notna()
    s, g = s[ok], g[ok]
    cutoff = s.quantile(cutoff_quantile)
    selected = (s >= cutoff).astype(int)
    df = pd.DataFrame({"group": g.to_numpy(), "selected": selected.to_numpy()}).groupby("group").agg(n=("selected", "size"), selection_rate=("selected", "mean")).reset_index()
    ref = _reference(g, reference)
    ref_rate = float(df.loc[df["group"] == ref, "selection_rate"].iloc[0])
    df["reference"] = ref
    df["AIR"] = df["selection_rate"] / ref_rate if ref_rate > 0 else np.nan
    df["flag"] = df["AIR"].round(10) < AIR_FLAG_THRESHOLD  # rounding keeps 0.08/0.10 at exactly 0.8
    return df.sort_values("n", ascending=False).reset_index(drop=True)


def standardized_mean_difference(score, group, reference: str = "largest") -> pd.DataFrame:
    """SMD of the score for each group versus the reference group (pooled SD)."""
    s = pd.Series(np.asarray(score, dtype=float)).reset_index(drop=True)
    g = _groups(group)
    ok = s.notna() & g.notna()
    s, g = s[ok], g[ok]
    ref = _reference(g, reference)
    ref_vals = s[g == ref]
    records = []
    for name, vals in s.groupby(g):
        pooled = np.sqrt((vals.var(ddof=1) * (len(vals) - 1) + ref_vals.var(ddof=1) * (len(ref_vals) - 1)) / max(len(vals) + len(ref_vals) - 2, 1))
        records.append({"group": name, "n": int(len(vals)), "mean": float(vals.mean()), "reference": ref, "smd": float((vals.mean() - ref_vals.mean()) / pooled) if pooled > 0 else 0.0})
    return pd.DataFrame(records).sort_values("n", ascending=False).reset_index(drop=True)


def proxy_test(score, protected_df: pd.DataFrame) -> dict:
    """OLS of the score on protected attributes; returns ``r2``, ``n`` and a coefficient table with 95% CIs.

    String / category columns of ``protected_df`` enter as ``C(col)``, numeric ones linearly.
    """
    df = protected_df.reset_index(drop=True).copy()
    df["score"] = np.asarray(score, dtype=float)
    terms = []
    for col in protected_df.columns:
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            terms.append(f"Q('{col}')")
        else:
            df[col] = df[col].astype("str")
            terms.append(f"C(Q('{col}'))")
    df = df.dropna()
    fit = smf.ols("score ~ " + " + ".join(terms), data=df).fit()
    ci = fit.conf_int()
    coefs = pd.DataFrame({"term": fit.params.index, "coef": fit.params.to_numpy(), "lo": ci[0].to_numpy(), "hi": ci[1].to_numpy(), "pvalue": fit.pvalues.to_numpy()})
    coefs["term"] = coefs["term"].str.replace("Q('", "", regex=False).str.replace("')", "", regex=False)
    log.info("proxy_test: R2=%.4f on n=%d", fit.rsquared, int(fit.nobs))
    return {"r2": float(fit.rsquared), "n": int(fit.nobs), "coefficients": coefs}


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V between two categorical series (NA pairs dropped)."""
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table.to_numpy(), correction=False)[0]
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def _is_categorical(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(s.dtype)


def _binned(s: pd.Series, n_bins: int = 10) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    if v.nunique(dropna=True) <= n_bins:
        return v.astype("str").mask(v.isna())
    return pd.qcut(v.rank(method="first"), n_bins, labels=False).astype("Float64").astype("str").mask(v.isna())


def feature_proxy_screen(X: pd.DataFrame, protected_df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """Association of each feature with each protected attribute; ``flag`` when above ``threshold``."""
    records = []
    P = protected_df.reset_index(drop=True)
    Xr = X.reset_index(drop=True)
    for feat in Xr.columns:
        for attr in P.columns:
            f, p = Xr[feat], P[attr]
            if _is_categorical(f) or _is_categorical(p):
                fa = f.astype("str").mask(f.isna()) if _is_categorical(f) else _binned(f)
                pa = p.astype("str").mask(p.isna()) if _is_categorical(p) else _binned(p)
                ok = fa.notna() & pa.notna()
                stat, method = cramers_v(fa[ok], pa[ok]), "cramers_v"
            else:
                ok = f.notna() & p.notna()
                stat = abs(float(spearmanr(f[ok], p[ok]).statistic)) if ok.sum() > 2 else 0.0
                method = "abs_spearman"
            records.append({"feature": feat, "attribute": attr, "stat": stat, "method": method, "flag": stat > threshold})
    out = pd.DataFrame(records)
    log.info("feature_proxy_screen: %d/%d feature-attribute pairs flagged at %.2f", int(out["flag"].sum()), len(out), threshold)
    return out


@dataclass
class LDAResult:
    """Outcome of the less-discriminatory-alternative search."""

    table: pd.DataFrame
    dropped: list[str]
    model: object | None
    score_test: np.ndarray | None


def _fairness_summary(label: str, n_features: int, y_test, score, group, cutoff_quantile: float) -> dict:
    air = adverse_impact_ratio(score, group, cutoff_quantile)
    smd = standardized_mean_difference(score, group)
    return {
        "variant": label, "n_features": n_features, "auc": auc(y_test, score), "min_AIR": float(air["AIR"].min()),
        "n_groups_flagged": int(air["flag"].sum()), "max_abs_SMD": float(smd["smd"].abs().max()),
    }


def less_discriminatory_alternative(
    model_factory: Callable[[], object],
    X_train: pd.DataFrame, y_train, X_valid: pd.DataFrame, y_valid, X_test: pd.DataFrame, y_test,
    group_test, flagged_features: Sequence[str], baseline_score: np.ndarray, cutoff_quantile: float = 0.9,
) -> LDAResult:
    """Refit without the flagged features and compare AUC / min AIR / max |SMD| with the baseline.

    When nothing is flagged the table carries the baseline only and ``model`` is ``None``.
    """
    base = _fairness_summary("baseline", X_train.shape[1], y_test, baseline_score, group_test, cutoff_quantile)
    dropped = [f for f in flagged_features if f in X_train.columns]
    if not dropped:
        log.info("less_discriminatory_alternative: no flagged features to drop")
        return LDAResult(pd.DataFrame([base]), [], None, None)
    keep = [c for c in X_train.columns if c not in dropped]
    model = model_factory()
    model.fit(X_train[keep], y_train, X_valid[keep], y_valid)
    score = model.predict_proba(X_test[keep])
    alt = _fairness_summary("drop_flagged", len(keep), y_test, score, group_test, cutoff_quantile)
    log.info("less_discriminatory_alternative: dropped %s; AUC %.4f -> %.4f, min AIR %.3f -> %.3f", dropped, base["auc"], alt["auc"], base["min_AIR"], alt["min_AIR"])
    return LDAResult(pd.DataFrame([base, alt]), dropped, model, score)

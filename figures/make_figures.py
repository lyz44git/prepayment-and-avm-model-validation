"""Figures 1 to 9 of the README, drawn from the tables in reports/ with the helpers in figures/figstyle.py.

Usage: python figures/make_figures.py   (writes figures/figure0 to figure9 as .png and .svg, and *-dark.svg for dark mode)
"""
import math
import re
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import *  # noqa: E402,F401,F403

DIP = pd.Timestamp("2024-09-15")          # the September 2024 rate dip


def csv(rel):
    return pd.read_csv(ROOT / rel)


def month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def row_axis(ax, labels, xlabel):
    ax.set_yticks(range(len(labels)), labels[::-1])
    ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False); clean(ax, x=False)
    ax.set_xlabel(xlabel)
    for t in ax.get_yticklabels():
        t.set_color(INK); t.set_fontsize(8)


def legend(ax, items, **kw):
    """items: (label, color, kind) with kind 'dot', 'diamond', 'open', 'line', 'dash' or 'bar'."""
    h = []
    for lab, c, kind in items:
        if kind == "bar":
            h.append(Patch(color=c, label=lab, lw=0))
        elif kind in ("line", "dash"):
            h.append(Line2D([], [], color=c, lw=2, ls="-" if kind == "line" else (0, (4, 3)), label=lab))
        else:
            h.append(Line2D([], [], ls="", marker="o" if kind == "dot" else "D", ms=6, mfc=c if kind != "open" else BG,
                            mec=c if kind == "open" else BG, mew=1.5 if kind == "open" else 1, label=lab))
    kw.setdefault("handlelength", 1.4); kw.setdefault("borderaxespad", 0.2)
    leg = ax.legend(handles=h, **kw)
    for t in leg.get_texts():
        t.set_color(INK2)
    return leg


# ------------------------------------------------------------------ prepayment model
def p_auc(ax, letter, title):
    """Sale-hazard AUC of the logit champion and the LightGBM challenger, in sample and out of time."""
    e = csv(P1 + "eval_monthly_hazard_final.csv").set_index(["cause", "split", "model"]).auc
    rows_ = [("In sample", "train"), ("Out of time", "oot"), ("Hold-out states,\nout of time", "spatial_oot")]
    for i, (_, sp) in enumerate(rows_):
        y = len(rows_) - 1 - i
        a, b = e[("sale", sp, "m1")], e[("sale", sp, "m2")]
        big = b - a > 0.01
        pill(ax, 0.6, 0.72, y, PALE, lw=7, z=1, alpha=0.7)
        pill(ax, min(a, b), max(a, b), y, ORANGE if big else GREY, lw=7, z=2, alpha=0.75)
        dot(ax, a, y, NAVY, s=46, z=4)
        ax.scatter([b], [y], s=80, marker="D", facecolors="none", edgecolors=BLUE, linewidths=1.8, zorder=5)
        gap = f"{b - a:+.3f}" if abs(b - a) >= 0.0005 else "0.000"
        ax.text(0.727, y, gap, va="center", fontsize=8.5, fontweight="bold", color=ORANGE if big else MUTED)
    ax.text(0.727, 2.45, "gap", fontsize=7.2, color=MUTED)
    ax.set_xlim(0.6, 0.75); ax.set_ylim(-0.6, 2.9)
    row_axis(ax, [r[0] for r in rows_], "AUC, monthly sale hazard")
    ax.set_xticks([0.60, 0.65, 0.70])
    legend(ax, [("Logit (champion)", NAVY, "dot"), ("LightGBM", BLUE, "open")], loc="upper left", ncol=2, columnspacing=1.0,
           bbox_to_anchor=(0.0, 1.02))
    panel(ax, letter, title)


def p_cpr(ax, letter, title):
    """Out-of-time pool CPR: actual, the logit fixed at 2024-06, and the logit with trailing recalibration."""
    m = csv(P1 + "monthly_cpr_actual_vs_pred.csv"); m = m[m.split == "oot"].copy(); m["month"] = pd.to_datetime(m.month)
    t = csv(P1 + "ts_benchmark_paths.csv"); t["month"] = pd.to_datetime(t.month)
    rec = t["hazard: logit + rolling 3-month recalibration | rolling 1-month-ahead"]
    b = csv(P1 + "ts_benchmark.csv").set_index(["model", "mode"]).MAE_pp
    mae_fixed = b[("hazard: logit champion (M1)", "static")]
    mae_rec = b[("hazard: logit + rolling 3-month recalibration", "rolling 1-month-ahead")]
    ax.plot(m.month, m.cpr_total_m1 * 100, color=GREY, lw=1.8, ls=(0, (4, 3)), zorder=3)
    ax.plot(t.month, rec, color=BLUE, lw=2.0, zorder=3)
    ax.plot(m.month, m.cpr_total_y * 100, color=INK, lw=2.2, marker="o", ms=4, mec=BG, mew=1, zorder=4)
    ax.axvline(DIP, color=ORANGE, lw=1, ls=(0, (2, 2)))
    ax.text(DIP + pd.Timedelta(days=6), 2.3, "rate dip", color=ORANGE, fontsize=7.5, fontweight="bold")
    legend(ax, [("Actual", INK, "line"), (f"Logit, recalibrated (error {mae_rec:.2f})", BLUE, "line"),
                (f"Logit, fixed at 2024-06 (error {mae_fixed:.2f})", GREY, "dash")], loc="upper center", bbox_to_anchor=(0.55, 1.03))
    ax.set_ylim(2.2, 5.9); month_axis(ax)
    ax.set_ylabel("Pool CPR (%)"); clean(ax)
    panel(ax, letter, title)


def p_avm(ax, letter, title):
    a = csv(P2 + "accuracy_oot_common.csv").set_index("estimator")
    n = int(a.loc["C3 blend (C1+C2)", "n"]); a = a.MdAPE * 100
    items = [("Vendor A AVM", "Vendor A AVM (champion)", GREY), ("Vendor A market value", "Vendor A market value", GREY),
             ("Hedonic OLS", "B1 hedonic OLS", LIGHT), ("LightGBM", "C1 LightGBM", BLUE),
             ("Attention network", "C2 neighbour-attention NN", BLUE), ("Blend", "C3 blend (C1+C2)", NAVY)]
    hbars(ax, [i[0] for i in items], [a[i[1]] for i in items], [i[2] for i in items], lambda v: f"{v:.1f}%", vmax=40)
    ax.set_xlabel(f"Median abs. % error, the same {n:,} later sales", color=MUTED)
    panel(ax, letter, title)


def block_ci(d, pred, act, block=3, B=2000, seed=0):
    """95% moving-block bootstrap interval (blocks of months) for sum(pred) / sum(act)."""
    rng = np.random.default_rng(seed); pv, av = d[pred].to_numpy(), d[act].to_numpy(); n = len(pv)
    starts = np.arange(n - block + 1); k = int(np.ceil(n / block)); out = np.empty(B)
    for b in range(B):
        idx = (rng.choice(starts, k)[:, None] + np.arange(block)).ravel()[:n]
        out[b] = pv[idx].sum() / av[idx].sum()
    return np.percentile(out, [2.5, 97.5])


def p_ratio(ax, letter, title):
    """Predicted / actual hazard of the logit, by split and cause, with month-block bootstrap intervals."""
    e = csv(P1 + "eval_monthly_hazard_final.csv").set_index(["cause", "split", "model"]).pred_over_actual
    mo = csv(P1 + "r4_f2_by_month.csv")
    for c in ("sale", "new_lien"):
        mo[f"pred_{c}"] = mo[f"pred_hazard_{c}_published"] * mo.property_months
    splits = [("In sample", "train", "train_states", "training months"), ("Out of time", "oot", "train_states", "out of time"),
              ("Hold-out states", "spatial", "holdout_states", "training months"),
              ("Hold-out, out of time", "spatial_oot", "holdout_states", "out of time")]
    ax.axvline(1, color=MUTED, lw=0.9, zorder=1)
    for i, (_, sp, geo, win) in enumerate(splits):
        y = len(splits) - 1 - i; d = mo[(mo.geo == geo) & (mo.window_published == win)]
        for c, col, dy, sz in (("sale", NAVY, 0.13, 70), ("new_lien", ORANGE, -0.13, 52)):
            v = e[(c, sp, "m1")]; lo, hi = block_ci(d, f"pred_{c}", f"actual_{c}")
            assert abs(d[f"pred_{c}"].sum() / d[f"actual_{c}"].sum() - v) < 0.002
            ax.plot([lo, hi], [y + dy, y + dy], color=col, lw=2.2, solid_capstyle="round", zorder=3, alpha=0.85)
            dot(ax, v, y + dy, col, s=sz, z=5)
            if c == "new_lien":
                ax.text(lo - 0.015, y + dy, f"{v:.2f}", ha="right", va="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_xlim(0.45, 1.12); ax.set_ylim(-0.6, 4.2)
    row_axis(ax, [s[0] for s in splits], "Predicted ÷ actual hazard (logit), 95% month-block interval")
    legend(ax, [("Sale", NAVY, "dot"), ("New first lien", ORANGE, "dot")], loc="upper left", ncol=2, columnspacing=1.0,
           bbox_to_anchor=(0.0, 1.02))
    panel(ax, letter, title)


def p_calib(ax, letter, title):
    c = csv(P1 + "cum12_calibration_oot_m1.csv")
    for _, r in c.iterrows():
        d = int(r.dec) + 1
        ax.plot([d, d], [r.pred * 100, r.act * 100], color=OSOFT, lw=4, solid_capstyle="round", zorder=1)
        dot(ax, d, r.act * 100, NAVY, s=70, z=4); dot(ax, d, r.pred * 100, GREY, s=30, z=5)
    legend(ax, [("Actual", NAVY, "dot"), ("Predicted", GREY, "dot")], loc="upper left")
    ax.set_xticks(range(1, 11)); ax.set_xlim(0.4, 10.6); ax.set_ylim(1.5, 6.2)
    ax.set_xlabel("Decile of predicted 12-month risk"); ax.set_ylabel("12-month termination rate (%)"); clean(ax)
    panel(ax, letter, title)


def p_or(ax, letter, title):
    s = csv(P1 + "sensitivity_lockin.csv")
    names = {"baseline (purchase-month PMMS)": "Baseline", "tenure knots (12,60,120,240)": "Other tenure knots",
             "stability: 2022-05..2023-06 only": "2022-05 to 2023-06 only", "stability: 2023-07..2024-06 only": "2023-07 to 2024-06 only",
             "case-control seed 1": "Another sampling seed",
             "note rate = cohort median recorded rate (regression calibration)": "Cohort rate proxy"}
    rows_ = []
    for cause, color, tag in [("new_lien", NAVY, "New lien: "), ("sale", GREY, "Sale: ")]:
        for spec, nm in names.items():
            r = s[(s.cause == cause) & (s.spec == spec)]
            if r.empty or (cause == "sale" and nm not in ("Baseline", "Other tenure knots")):
                continue
            b, se = float(r.gap_pos.iloc[0]), float(r.se.iloc[0])
            rows_.append((tag + nm, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), color))
    n = len(rows_)
    ax.axvline(1.0, color=ORANGE, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(1.004, n - 0.35, "no effect", fontsize=7.2, color=ORANGE, fontweight="bold", va="bottom")
    for i, (lab, est, lo, hi, c) in enumerate(rows_):
        y = n - 1 - i
        ax.plot([lo, hi], [y, y], color=c, lw=2.2, zorder=3, solid_capstyle="round")
        dot(ax, est, y, c, s=48)
        ax.text(1.115, y, f"{est:.2f}", va="center", fontsize=8, fontweight="bold", color=INK)
    ax.set_xlim(0.8, 1.15); ax.set_ylim(-0.6, n + 0.1); ax.set_xticks([0.8, 0.9, 1.0, 1.1])
    row_axis(ax, [r[0] for r in rows_], "Odds ratio per +1 pp rate gap (95% CI)")
    panel(ax, letter, title)


def p_stress(ax, letter, title):
    s = csv(P1 + "stress_summary.csv").set_index(["scenario", "model"]).delta_cpr_count_vs_base_pp
    meta = load(P1 + "stress_meta.json"); lo_r, hi_r = meta["train_pmms_range"]; r0 = meta["pmms_2025_06"]
    sc = [("−200 bp", "-200bp", r0 - 2), ("−100 bp", "-100bp", r0 - 1), ("Realised rates", "realized PMMS path", None),
          ("+100 bp", "+100bp", r0 + 1), ("+200 bp", "+200bp", r0 + 2)]
    for i, (_, key, rate) in enumerate(sc):
        y = len(sc) - 1 - i
        if rate is not None and not (lo_r <= rate <= hi_r):
            ax.axhspan(y - 0.47, y + 0.47, color=OSOFT, alpha=0.5, lw=0, zorder=0)
        for m, c, off in [("m1", NAVY, 0.17), ("m2", LIGHT, -0.17)]:
            v = s[(key, m)]
            ax.barh(y + off, v, height=0.32, color=c, zorder=2, lw=0)
            ax.text(v + (0.03 if v >= 0 else -0.03), y + off, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=7.5, color=INK if m == "m1" else MUTED, fontweight="bold" if m == "m1" else "normal")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.text(1.38, 0.5, f"Shaded: rates outside the\ntraining range ({lo_r:.1f}–{hi_r:.1f}%)", ha="right", va="center", fontsize=7,
            color=ORANGE, fontweight="bold")
    ax.set_xlim(-1.1, 1.42); ax.set_ylim(-0.6, len(sc) - 0.4)
    row_axis(ax, [x[0] for x in sc], "Change in 24-month CPR vs. base (pp)")
    legend(ax, [("Logit", NAVY, "bar"), ("LightGBM", LIGHT, "bar")], loc="center left", bbox_to_anchor=(0.0, 0.66))
    panel(ax, letter, title)


def p_hpi(ax, letter, title):
    s = csv(REM + "step5_stress_summary.csv").set_index(["scenario", "spec"]).delta_cpr_count_vs_base_pp
    sc = [("−20%", "HPI -20% (rates flat)"), ("−10%", "HPI -10% (rates flat)"), ("+10%", "HPI +10% (rates flat)")]
    for i, (_, key) in enumerate(sc):
        y = len(sc) - 1 - i
        for spec, c, off in [("S0", GREY, 0.17), ("S1", BLUE, -0.17)]:
            v = s[(key, spec)]
            ax.barh(y + off, v, height=0.32, color=c, zorder=2, lw=0)
            ax.text(v + (0.0015 if v >= 0 else -0.0015), y + off, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=7.5, color=INK if spec == "S1" else MUTED, fontweight="bold" if spec == "S1" else "normal")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(-0.05, color=ORANGE, lw=1.1, ls=(0, (4, 3)))
    ax.text(-0.0492, 2.47, "closure bar\nfor −10%: −0.05", color=ORANGE, fontsize=7, fontweight="bold", va="center")
    ax.set_xlim(-0.062, 0.03); ax.set_ylim(-0.6, len(sc) - 0.3); ax.set_xticks([-0.06, -0.04, -0.02, 0, 0.02])
    row_axis(ax, [x[0] for x in sc], "Change in 24-month CPR vs. base (pp)")
    ax.set_ylabel("House-price shock", color=INK2)
    legend(ax, [("Published model", GREY, "bar"), ("Remediated: LTV only", BLUE, "bar")], loc="lower left",
           bbox_to_anchor=(0.1, 0.0))
    panel(ax, letter, title)


def p_ltv(ax, letter, title):
    d = csv(REM + "diag_hazard_by_ltv_band_train.csv")
    labels = [re.sub(r"\[(\d\.\d), (\d\.\d)\)", lambda m: f"{m.group(1)}–{m.group(2)}", b) for b in d.band]
    vals = list(d.share_of_rows * 100)
    cols = [ORANGE if float(l[:3]) >= 0.8 else (BLUE if v > 10 else LIGHT) for l, v in zip(labels, vals)]
    note = [""] * (len(vals) - 1) + ["where a price fall would bite"]
    hbars(ax, labels, vals, cols, lambda v: f"{v:.1f}%" if v >= 0.1 else f"{v:.3f}%", vmax=50, note=note)
    ax.set_ylabel("Estimated LTV", color=INK2); ax.set_xlabel("Share of training property-months", color=MUTED)
    panel(ax, letter, title)


def p_nl_month(ax, letter, title):
    w = csv(REM + "step4_walkforward_by_month.csv")
    w = w[(w.spec == "S1") & (w.split == "oot") & (w.cause == "new_lien")].copy(); w["month"] = pd.to_datetime(w.month)
    ax.axhspan(0.9, 1.1, color=PALE, alpha=0.7, lw=0, zorder=0); ax.axhline(1, color=MUTED, lw=0.8)
    ax.text(w.month.iloc[-1], 1.05, "closure range, pooled re-estimate: 0.90–1.10", color=MUTED, fontsize=7.2, ha="right", va="center")
    ax.plot(w.month, w.fixed_origin / w.actual_hazard, color=GREY, lw=1.8, ls=(0, (4, 3)), marker="o", ms=3.5, mec=BG, zorder=3)
    ax.plot(w.month, w.walk_forward / w.actual_hazard, color=BLUE, lw=2.2, marker="o", ms=4.5, mec=BG, mew=1, zorder=4)
    ax.axvline(DIP, color=ORANGE, lw=1, ls=(0, (2, 2)))
    ax.text(DIP + pd.Timedelta(days=6), 0.52, "rate dip", color=ORANGE, fontsize=7.5, fontweight="bold")
    legend(ax, [("Re-estimated each month", BLUE, "line"), ("Fixed at 2024-06", GREY, "dash")], loc="lower right")
    ax.set_ylim(0.5, 1.15); month_axis(ax)
    ax.set_ylabel("Predicted ÷ actual\nnew-lien hazard"); clean(ax)
    panel(ax, letter, title)


def p_mae(ax, letter, title):
    t = csv(P1 + "ts_benchmark.csv").set_index(["model", "mode"]).MAE_pp
    c = csv(REM + "step3_4_cpr_backtest.csv").set_index(["model", "split"]).MAE_monthly_cpr_pp
    items = [("Logit, fixed at 2024-06", t[("hazard: logit champion (M1)", "static")], GREY),
             ("Logit, re-estimated monthly", c[("S1 walk-forward", "oot")], GREY),
             ("Flat forecast", t[("naive", "static")], LIGHT),
             ("Last month's CPR (rolling)", t[("naive", "rolling 1-month-ahead")], LIGHT),
             ("Rolling ARX(1) with PMMS", t[("ARX(1) + PMMS", "rolling 1-month-ahead")], LIGHT),
             ("Logit + trailing recalibration", t[("hazard: logit + rolling 3-month recalibration", "rolling 1-month-ahead")], NAVY)]
    hbars(ax, [i[0] for i in items], [i[1] for i in items], [i[2] for i in items], lambda v: f"{v:.2f}", vmax=0.55)
    ax.set_xlabel("Mean absolute error of monthly pool CPR, out of time (pp)", color=MUTED)
    panel(ax, letter, title)


# ------------------------------------------------------------------ valuation study
def p_vendor_q(ax, letter, title):
    q = csv(P2 + "vendor_accuracy_by_quarter.csv")
    order = q.group.drop_duplicates().tolist(); x = np.arange(len(order))
    spec = [("Vendor B est. home value", ORANGE, "Vendor B", 0.0), ("Vendor A market value", BLUE, "Vendor A market value", 0.012),
            ("Assessor value (Vendor C 2025)", GREY, "Assessor value", -0.012), ("Vendor A AVM", NAVY, "Vendor A AVM", 0.0)]
    for est, c, lab, dy in spec:
        r = q[q.estimator == est].set_index("group").reindex(order).median_ratio
        ax.plot(x, r, color=c, lw=2.0, marker="o", ms=3.5, mec=BG, zorder=3)
        if est.startswith("Vendor B"):
            k = order.index("2024Q3")
            ax.text(k / 2, 1.025, "Vendor B median value ÷ price = 1.00 through 2024Q3", color=ORANGE, fontsize=7.2, fontweight="bold",
                    ha="center")
            continue
        ax.text(x[-1] + 0.4, r.iloc[-1] + dy, lab, color=c, fontsize=7.2, fontweight="bold", va="center")
    ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ticks = [i for i, g in enumerate(order) if g.endswith("Q1")]
    ax.set_xticks(ticks, [order[i][:4] for i in ticks]); ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0.6, 1.07); ax.set_ylabel("Median value ÷ sale price"); ax.set_xlabel("Quarter of sale"); clean(ax)
    panel(ax, letter, title)


def p_geo(ax, letter, title):
    """Blend minus the best usable vendor value on the same sales, with county-cluster bootstrap intervals."""
    b = load(P2 + "r4_bootstrap_b1000.json")
    est0 = b["point_estimate_full_sample"]["delta_MdAPE_challenger_minus_mktval"]; lo0, _, hi0 = b["bootstrap_B1000"]["delta_MdAPE_challenger_minus_mktval"]
    sp = csv(P2 + "r4_accuracy_spatial_oot.csv").drop_duplicates("window").set_index("window")
    w = {k: sp.loc[[i for i in sp.index if i.startswith(k)][0]] for k in ("all months", "2022-05", "2024-07")}
    rows_ = [(f"Out of time,\n{b['n'] / 1000:.0f}K sales", est0, lo0, hi0)] + [
        (lab, w[k].c3_minus_mktval_MdAPE_common, w[k].c3_minus_mktval_boot_ci95_lo, w[k].c3_minus_mktval_boot_ci95_hi)
        for lab, k in (("Unseen states,\nall months", "all months"), ("Unseen states,\nbefore Jul 2024", "2022-05"),
                       ("Unseen states,\nout of time", "2024-07"))]
    est = [100 * r[1] for r in rows_]; lo = [100 * r[2] for r in rows_]; hi = [100 * r[3] for r in rows_]
    cols = [NAVY if h < 0 else (ORANGE if l > 0 else GREY) for l, h in zip(lo, hi)]
    dotrows(ax, [r[0] for r in rows_], est, lo, hi, cols, lambda v: f"{v:+.1f}".replace("-", "−"), (-9.5, 6.5), ref=0,
            ref_label="no difference", pad=0.025)
    ax.text(-9.3, -0.55, "← blend better", fontsize=7, color=MUTED, va="center"); ax.text(6.3, -0.55, "vendor better →", fontsize=7, color=MUTED, va="center", ha="right")
    ax.set_xlabel("Blend minus vendor, median abs. error (pp), same sales; 95% CI")
    panel(ax, letter, title)


def p_drift(ax, letter, title):
    m = csv(P2 + "r4_monitoring_backtest.csv"); cal = m[m.window.str.startswith("calibration")].iloc[0]
    o = m[m.window == "out of time"].copy(); o["month"] = pd.to_datetime(o.month)
    ps = csv(P2 + "psi_backfill_summary.csv"); ps = ps[ps.split == "oot"]
    psi_max = math.ceil(max(ps.prediction_psi.max(), ps.max_unexpected_psi.max()) * 1000) / 1000
    thr = 100 * cal.MdAPE_C1_LightGBM + o.trigger_threshold_pp.iloc[0]
    y = o.MdAPE_C1_LightGBM * 100
    ax.axhline(thr, color=ORANGE, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(o.month.iloc[0], thr + 0.06, f"trigger: calibration error + {o.trigger_threshold_pp.iloc[0]:.1f} pt", fontsize=7, color=ORANGE,
            fontweight="bold", va="bottom")
    ax.plot(o.month, y, color=NAVY, lw=2.2, marker="o", ms=4, mec=BG, mew=1, zorder=3)
    first = o[o.C1_trigger_single_month].month.iloc[0]; fire = o[o.C1_trigger_two_consecutive].month.iloc[0]
    dot(ax, fire, y[o.month == fire].iloc[0], ORANGE, s=70, z=5)
    ax.annotate(f"crosses {first:%b}, fires {fire:%b %Y}", (fire, y[o.month == fire].iloc[0]), xytext=(-8, 22), textcoords="offset points",
                ha="right", fontsize=7.2, color=ORANGE, fontweight="bold", arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    ax.text(o.month.iloc[0] + pd.Timedelta(days=12), y.iloc[0] - 0.2, f"{y.iloc[0]:.1f}%", fontsize=8, color=INK, fontweight="bold", va="top")
    ax.text(o.month.iloc[-1], y.iloc[-1] + 0.25, f"{y.iloc[-1]:.1f}%", fontsize=8, color=INK, fontweight="bold", ha="center")
    ax.text(o.month.iloc[-1], 10.35, f"Prediction and stable-input PSI\n≤ {psi_max:.3f} in all {len(ps)} months", fontsize=7.3, color=INK2,
            ha="right", va="center", bbox=dict(boxstyle="round,pad=0.45", fc=PALE, ec="none"))
    ax.set_ylim(10, 13.6); month_axis(ax); ax.set_ylabel("Median absolute % error, LightGBM"); clean(ax)
    panel(ax, letter, title)


# ------------------------------------------------------------------ credit-loss study (reports/p3)
LOSS_MODELS = [("M0", "Lookup", GREY), ("M1", "Logit (champion)", NAVY), ("M1r", "Logit, recalibrated", LIGHT), ("M1v", "Logit + approval eras", BLUE)]


def log_ratio_axis(ax, lo=0.25, hi=10):
    ax.set_yscale("log"); ax.set_ylim(lo, hi)
    ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8) if lo <= t <= hi]
    ax.set_yticks(ticks, [f"{t:g}×" for t in ticks]); ax.minorticks_off()
    ax.axhspan(0.75, 1.25, color=PALE, alpha=0.8, lw=0, zorder=0); ax.axhline(1, color=MUTED, lw=0.8, zorder=1)


def p_bt_overview(ax, letter, title):
    b = csv(P3 + "backtest_rolling.csv"); b = b[b.macro == "realised"]
    org = b.origin.drop_duplicates().tolist(); x = np.arange(len(org))
    for key, lab, c in [("M1", "Logit (champion)", NAVY), ("M1v", "Logit + approval eras", BLUE)]:
        r = b[b.model == key].set_index("origin").reindex(org).dollar_ratio
        ax.plot(x, r, color=c, lw=0.9, alpha=0.4, zorder=2, ls="-" if key == "M1" else (0, (4, 2)))   # guide only: origins overlap
        ax.scatter(x, r, s=44 if key == "M1" else 36, color=c, edgecolor=BG, linewidth=1, zorder=3, label=lab)
    log_ratio_axis(ax)
    k = org.index("2019-12"); v = b[(b.model == "M1") & (b.origin == "2019-12")].dollar_ratio.iloc[0]
    ax.annotate(f"{v:.1f}× from Dec 2019 (payment relief)", (k, v), xytext=(8, -2), textcoords="offset points", ha="left", va="center", fontsize=7.3, color=ORANGE, fontweight="bold")
    ax.text(len(org) - 0.35, 1.27, "±25%", fontsize=7, color=MUTED, va="bottom", ha="right")
    ax.set_xticks(x, [o[:4] for o in org]); ax.set_xlim(-0.4, len(org) - 0.3)
    ax.set_xlabel("Forecast origin (December)"); ax.set_ylabel("24-month charge-off $: predicted ÷ actual"); clean(ax)
    legend(ax, [("Champion logit", NAVY, "dot"), ("+ approval eras, no overlay", BLUE, "dot")],
           loc="upper left", fontsize=7)
    panel(ax, letter, title)


def p_term_field(ax, letter, title):
    t = csv(P3 + "term_field_check.csv").set_index("final_status")
    order = [("Cancelled before\ndisbursement", "cancelled (never disbursed)", GREY), ("Active", "active", LIGHT),
             ("Paid in full", "paid in full", LIGHT), ("Charged off", "charged off", ORANGE)]
    hbars(ax, [o[0] for o in order], [100 * t.loc[o[1], "term_not_multiple_of_12"] for o in order], [o[2] for o in order],
          lambda v: f"{v:.0f}%", vmax=100)
    ax.set_xlabel("Loans whose recorded term is not a multiple of 12 months", color=MUTED)
    panel(ax, letter, title)


def p_leak_auc(ax, letter, title):
    old = csv(P3 + "first_fit/eval_hazard.csv").query("cause == 'co'").set_index(["split", "model"]).auc
    new = csv(P3 + "eval_hazard.csv").query("cause == 'co'").set_index(["split", "model"]).auc
    rows_ = [("LightGBM, in sample", "in sample 2006-19", "M2 LightGBM"), ("Logit, in sample", "in sample 2006-19", "M1 logit"),
             ("LightGBM, out of time", "out of time 2020-26", "M2 LightGBM"), ("Logit, out of time", "out of time 2020-26", "M1 logit")]
    for i, (lab, sp, md) in enumerate(rows_):
        y = len(rows_) - 1 - i
        a, b_ = new[(sp, md)], old[(sp, md)]
        pill(ax, 0.6, 1.0, y, PALE, lw=7, z=1, alpha=0.7); pill(ax, a, b_, y, OSOFT, lw=7, z=2)
        dot(ax, a, y, NAVY, s=60, z=5); dot(ax, b_, y, ORANGE, s=60, z=4)
        ax.text(a - 0.012, y, f"{a:.2f}", ha="right", va="center", fontsize=7.8, fontweight="bold", color=INK)
        ax.text(b_ + 0.012, y, f"{b_:.2f}", ha="left", va="center", fontsize=7.8, color=ORANGE, fontweight="bold")
    ax.set_xlim(0.6, 1.03); ax.set_ylim(-0.6, 4.3)
    row_axis(ax, [r[0] for r in rows_], "AUC, monthly charge-off hazard")
    legend(ax, [("Term field removed", NAVY, "dot"), ("Revised term in the model", ORANGE, "dot")], loc="upper left", ncol=2,
           columnspacing=0.8, bbox_to_anchor=(0.0, 1.02))
    panel(ax, letter, title)


def p_vintage(ax, letter, title):
    v = csv(P3 + "vintage_pred_vs_actual.csv")
    v = v[(v.period == "in sample 2006-19") & v.approval_fy.between(2000, 2018)]
    cols = [ORANGE if 2006 <= fy <= 2008 else (BLUE if fy >= 2009 else GREY) for fy in v.approval_fy]
    ax.bar(v.approval_fy, v.m1_pred_over_actual, color=cols, width=0.72, zorder=2, lw=0)
    ax.axhline(1, color=MUTED, lw=0.8)
    r06 = v.set_index("approval_fy").m1_pred_over_actual
    top = r06.loc[2006:2008].max()
    ax.text(2007, top + 0.04, f"{r06[2007]:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=INK)
    ax.annotate("FY2006–08\nunder-predicted", (2007, top + 0.12), xytext=(2003.6, 1.42), ha="center", va="bottom", fontsize=7,
                color=ORANGE, fontweight="bold", linespacing=1.1, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    ax.text(r06.idxmax(), r06.max() + 0.05, f"{r06.max():.2f}", ha="center", fontsize=7.5, fontweight="bold", color=INK)
    ax.text(2013.5, 2.55, "FY2009–18 over-predicted", ha="center", fontsize=7, color=BLUE, fontweight="bold")
    ax.set_ylim(0, 2.75); ax.set_xticks([2000, 2004, 2008, 2012, 2016], ["FY2000", "2004", "2008", "2012", "2016"])
    ax.set_ylabel("Predicted ÷ actual charge-offs"); ax.set_xlabel("Approval fiscal year (in sample, 2006–2019)"); clean(ax)
    panel(ax, letter, title)


def p_backtest(ax, letter, title):
    b = csv(P3 + "backtest_rolling.csv"); b = b[b.macro == "realised"]
    org = b.origin.drop_duplicates().tolist(); x = np.arange(len(org))
    offs = dict(zip([m[0] for m in LOSS_MODELS], (-0.21, -0.07, 0.07, 0.21)))
    for key, lab, c in LOSS_MODELS:
        r = b[b.model == key].set_index("origin").reindex(org).dollar_ratio
        ax.scatter(x + offs[key], r, s=34 if key != "M1" else 44, color=c, edgecolor=BG, linewidth=0.8, zorder=4, label=lab)
    log_ratio_axis(ax)
    ax.set_xticks(x, [o[:4] for o in org]); ax.set_xlim(-0.5, len(org) - 0.5)
    ax.set_xlabel("Forecast origin (December)"); ax.set_ylabel("24-month charge-off $: predicted ÷ actual"); clean(ax)
    s = load(P3 + "backtest_summary.json")
    legend(ax, [(f"{lab}: {s[k]['within_25pct']}/8 within ±25%", c, "dot") for k, lab, c in LOSS_MODELS], loc="upper left", fontsize=7)
    panel(ax, letter, title)


def p_pandemic(ax, letter, title):
    mo = csv(P3 + "backtest_rolling_monthly.csv"); mo = mo[mo.origin == "2019-12"].copy(); mo["month"] = pd.to_datetime(mo.month)
    act = mo[(mo.model == "M1") & (mo.macro == "realised")]
    ax.axvspan(pd.Timestamp("2020-04-01"), pd.Timestamp("2021-12-31"), color=PALE, alpha=0.7, lw=0, zorder=0)
    ax.text(pd.Timestamp("2021-12-20"), 0.97, "payment-relief window", transform=ax.get_xaxis_transform(), fontsize=7, color=MUTED, va="top", ha="right")
    ax.plot(act.month, act.actual_dollars / 1e6, color=INK, lw=2.2, zorder=4)
    for key, macro, c, ls in [("M1", "realised", NAVY, "-"), ("M1", "held at origin", GREY, (0, (4, 3))), ("M1v", "realised", BLUE, (0, (2, 2)))]:
        r = mo[(mo.model == key) & (mo.macro == macro)]
        ax.plot(r.month, r.pred_dollars / 1e6, color=c, lw=1.8, ls=ls, zorder=3)
    legend(ax, [("Actual", INK, "line"), ("Logit, realised macro", NAVY, "line"), ("Logit, macro held at 2019-12", GREY, "dash"),
                ("Logit + eras, realised macro", BLUE, "dash")], loc="upper left", bbox_to_anchor=(0.0, 0.97), fontsize=7)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_ylabel("Charge-offs per month ($ million)"); clean(ax)
    panel(ax, letter, title)


def p_monitor(ax, letter, title):
    ps = csv(P3 + "backtest_psi.csv"); b = csv(P3 + "backtest_rolling.csv")
    r = b[(b.model == "M1") & (b.macro == "realised")].set_index("origin").dollar_ratio
    ps["ratio"] = ps["to"].map(r)
    ax.axhspan(0.8, 1.25, color=PALE, alpha=0.8, lw=0, zorder=0)
    ax.axvline(0.25, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(0.27, 9.0, "PSI 0.25:\nusual alarm", fontsize=7, color=MUTED, va="top")
    for _, q in ps.iterrows():
        hit = q.psi < 0.25 and q.ratio > 1.5
        dot(ax, q.psi, q.ratio, ORANGE if hit else NAVY, s=50)
        ax.annotate(q["to"][:4], (q.psi, q.ratio), xytext=(5, 3), textcoords="offset points", fontsize=7.3, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.005, 10); ax.set_ylim(0.4, 10)
    ax.set_xticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"]); ax.set_yticks([0.5, 1, 2, 4, 8], ["0.5×", "1×", "2×", "4×", "8×"]); ax.minorticks_off()
    ax.set_xlabel("PSI of the book's 24-month score vs. previous origin"); ax.set_ylabel("Logit forecast ÷ actual"); clean(ax)
    ax.grid(axis="x", color=GRID)
    panel(ax, letter, title)


def p_allowance(ax, letter, title):
    c = csv(P3 + "cecl_allowance.csv"); c = c[c.horizon == "recorded term"]
    order = [("M1v", "Logit + approval eras"), ("M1r", "Logit, recalibrated"), ("M1", "Logit (champion)"), ("M2", "LightGBM"), ("M0", "Lookup, no macro")]
    for i, (k, lab) in enumerate(order):
        y = len(order) - 1 - i
        bse = 100 * c[(c.model == k) & (c.scenario == "baseline")].loss_over_balance.iloc[0]
        sev = 100 * c[(c.model == k) & (c.scenario == "severe")].loss_over_balance.iloc[0]
        pill(ax, 0, 11, y, PALE, lw=7, z=1, alpha=0.6); pill(ax, bse, sev, y, OSOFT, lw=7, z=2)
        dot(ax, bse, y, NAVY, s=62, z=5); dot(ax, sev, y, ORANGE, s=62, z=4)
        ax.text(bse - 0.25, y, f"{bse:.1f}%", ha="right", va="center", fontsize=7.8, fontweight="bold", color=INK)
        if sev - bse > 0.05:
            ax.text(sev + 0.25, y, f"{sev:.1f}%", ha="left", va="center", fontsize=7.8, fontweight="bold", color=ORANGE)
    ax.set_xlim(0, 11.5); ax.set_ylim(-0.6, len(order) + 0.2)
    row_axis(ax, [o[1] for o in order], "Lifetime loss, % of estimated balance (active book, 2026-09)")
    legend(ax, [("Fed baseline", NAVY, "dot"), ("Fed severely adverse", ORANGE, "dot")], loc="upper right", ncol=2, columnspacing=0.8,
           bbox_to_anchor=(1.0, 1.02))
    panel(ax, letter, title)


def p_ur(ax, letter, title):
    m = csv(P3 + "cecl_macro_paths.csv"); m["month"] = pd.to_datetime(m.month)
    ax.plot(m.month, m.ur_baseline, color=NAVY, lw=2.0); ax.plot(m.month, m.ur_severe, color=ORANGE, lw=2.0)
    ax.axvline(pd.Timestamp("2026-09-30"), color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.text(pd.Timestamp("2026-10-15"), 10.6, "data date", fontsize=7, color=MUTED)
    ax.axvspan(pd.Timestamp("2029-04-01"), pd.Timestamp("2030-03-31"), color=PALE, alpha=0.7, lw=0, zorder=0)
    ax.text(pd.Timestamp("2029-04-20"), 3.3, "reversion", fontsize=7, color=MUTED)
    pub = csv(P3 + "macro_2026_scenario_vs_published.csv"); pub = pub[pub.source == "published_fred"]
    qd = pd.PeriodIndex(pub.quarter, freq="Q").to_timestamp(how="start") + pd.Timedelta(days=45)
    ax.scatter(qd, pub.unemployment, s=26, color=INK, zorder=5)
    ax.text(pd.Timestamp("2024-10-01"), 10.8, "dots:\npublished\n2026Q1–Q3;\nscenarios\nstart 2026Q1", fontsize=6.6, color=INK2,
            ha="left", va="top", linespacing=1.1)
    legend(ax, [("Baseline", NAVY, "line"), ("Severely adverse", ORANGE, "line"), ("Published", INK, "dot")], loc="upper right")
    ax.set_ylim(3, 11); ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Unemployment rate (%)"); clean(ax)
    panel(ax, letter, title)


def p_severity_year(ax, letter, title):
    y = csv(P3 + "annual_chargeoffs.csv"); y = y[y.year <= 2025]
    ax.plot(y.year, 100 * y.dollar_severity, color=NAVY, lw=2.0, marker="o", ms=4, mec=BG, zorder=3)
    for yr in (2010, 2019):
        v = 100 * y.set_index("year").dollar_severity[yr]
        ax.text(yr, v + 1.2, f"{v:.0f}%", ha="center", fontsize=7.8, fontweight="bold", color=INK)
    ax.set_ylim(40, 80); ax.set_xticks([2006, 2010, 2014, 2018, 2022])
    ax.set_ylabel("Charge-off $ ÷ original amount (%)"); ax.set_xlabel("Year of charge-off"); clean(ax)
    panel(ax, letter, title)


def p_backtest2(ax, letter, title):
    """24-month backtest: the champion before remediation against the remediated candidates (relief overlay O1)."""
    b0 = csv(P3 + "backtest_rolling.csv"); b0 = b0[(b0.macro == "realised") & (b0.model == "M1")]
    b2 = csv(P3 + "r2_backtest.csv"); b2 = b2[(b2.macro == "realised") & (b2.overlay)]
    org = b0.origin.tolist(); x = np.arange(len(org)); s = load(P3 + "r2_summary.json"); s0 = load(P3 + "backtest_summary.json")
    series = [("Champion, before", b0.set_index("origin").dollar_ratio, NAVY, s0["M1"]["within_25pct"]),
              ("Eras + relief overlay", b2[b2.model == "M1v"].set_index("origin").dollar_ratio, BLUE, s["M1v +O1"]["within_25pct"]),
              ("… + recalibration", b2[b2.model == "R2a"].set_index("origin").dollar_ratio, LIGHT, s["R2a +O1"]["within_25pct"])]
    for i, (lab, r, c, k) in enumerate(series):
        ax.scatter(x + (i - 1) * 0.17, r.reindex(org), s=40, color=c, edgecolor=BG, linewidth=0.8, zorder=4)
    log_ratio_axis(ax)
    ax.set_xticks(x, [o[:4] for o in org]); ax.set_xlim(-0.5, len(org) - 0.5)
    ax.set_xlabel("Forecast origin (December)"); ax.set_ylabel("24-month charge-off $: predicted ÷ actual"); clean(ax)
    legend(ax, [(f"{lab}: {k}/8 within ±25%", c, "dot") for lab, r, c, k in series], loc="upper left", fontsize=7)
    panel(ax, letter, title)


def p_pandemic2(ax, letter, title):
    mo = csv(P3 + "backtest_rolling_monthly.csv"); mo = mo[mo.origin == "2019-12"].copy(); mo["month"] = pd.to_datetime(mo.month)
    r2 = csv(P3 + "r2_backtest_monthly.csv"); r2 = r2[r2.overlay & (r2.model == "M1v")].copy(); r2["month"] = pd.to_datetime(r2.month)
    act = mo[(mo.model == "M1") & (mo.macro == "realised")]
    ax.axvspan(pd.Timestamp("2020-04-01"), pd.Timestamp("2021-09-30"), color=PALE, alpha=0.7, lw=0, zorder=0)
    ax.text(pd.Timestamp("2020-04-12"), 0.985, "SBA paid borrowers' loan payments", transform=ax.get_xaxis_transform(), fontsize=7,
            color=MUTED, va="top", ha="left")
    ax.plot(act.month, act.actual_dollars / 1e6, color=INK, lw=2.2, zorder=4)
    ax.plot(act.month, act.pred_dollars / 1e6, color=NAVY, lw=1.8, zorder=3)
    ax.plot(r2.month, r2.pred_dollars / 1e6, color=BLUE, lw=2.0, ls=(0, (4, 2)), zorder=3)
    legend(ax, [("Actual", INK, "line"), ("Champion, realised macro", NAVY, "line"), ("Eras + relief overlay (hindsight)", BLUE, "dash")],
           loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=7)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_ylabel("Charge-offs per month ($ million)"); clean(ax)
    panel(ax, letter, title)


def p_lifetime(ax, letter, title):
    lt = csv(P3 + "r3_lifetime_backtest.csv"); s = load(P3 + "r3_summary.json")
    org = lt.origin.drop_duplicates().tolist(); x = np.arange(len(org))
    series = [("M0", "Lookup", GREY, "o"), ("M1", "Champion", NAVY, "o"), ("M1v", "Eras + overlay", BLUE, "o"),
              ("R2a", "+ recalibration", LIGHT, "o"), ("R2b", "+ downturn exp.", BLUE, "D")]
    for i, (k, lab, c, mk) in enumerate(series):
        r = lt[lt.model == k].set_index("origin").reindex(org).dollar_ratio
        ax.scatter(x + (i - 2) * 0.13, r, s=34 if mk == "D" else 40, marker=mk, color=c, edgecolor=BG, linewidth=0.8, zorder=4)
    log_ratio_axis(ax, 0.25, 4)
    ax.set_xticks(x, [o[:4] for o in org]); ax.set_xlim(-0.5, len(org) - 0.5)
    ax.set_xlabel("Origin (December); outcomes to 2026-09"); ax.set_ylabel("Lifetime-to-date charge-off $: pred. ÷ actual"); clean(ax)
    legend(ax, [(f"{lab} {s[k]['within_25pct']}/6", c, "diamond" if mk == "D" else "dot") for k, lab, c, mk in series],
           loc="upper left", fontsize=6.8, title="within ±25% (rule 5/6)", title_fontsize=6.8, alignment="left",
           handletextpad=0.4, labelspacing=0.35)
    panel(ax, letter, title)


def p_allowance2(ax, letter, title):
    c2 = csv(P3 + "r2_cecl_allowance.csv").set_index(["scenario", "model"]).loss_over_balance
    c0 = csv(P3 + "cecl_allowance.csv"); c0 = c0[c0.horizon == "recorded term"].set_index(["scenario", "model"]).loss_over_balance
    sens = csv(P3 + "r4_allowance_sensitivities.csv"); sens = sens[sens.scenario == "baseline"].set_index(["variant", "model"]).loss_over_balance
    ALT = {("M1v", "Eras + relief overlay"): (100 * sens[("v", "M1v")], "FY2020+ loans keep the FY2014–19 effect"),
           ("R2a", "Eras + overlay, recalibrated"): (100 * sens[("iv", "R2a")], "factor phased out after 2029Q1")}
    order = [("M1v", "Eras + relief overlay", c2), ("R2b", "… + downturn exposure", c2), ("R2a", "Eras + overlay, recalibrated", c2),
             ("M1", "Champion, no overlay", c0), ("M1", "Champion, with overlay", c2), ("M2", "LightGBM (challenger)", c0),
             ("M0", "Lookup, no macro", c0)]
    for i, (k, lab, src) in enumerate(order):
        y = len(order) - 1 - i
        bse, sev = 100 * src[("baseline", k)], 100 * src[("severe", k)]
        pill(ax, 0, 11, y, PALE, lw=7, z=1, alpha=0.6); pill(ax, bse, sev, y, OSOFT, lw=7, z=2)
        db = ds = 0.0
        if 0.005 < abs(sev - bse) < 0.25:           # nearly coincident points: offset them vertically so both show
            db, ds = 0.14, -0.14
        dot(ax, bse, y + db, NAVY, s=58, z=5); dot(ax, sev, y + ds, ORANGE, s=58, z=4)
        left = sev < bse                             # severe can sit below baseline under the reversion of §9 (R2b)
        ax.text(bse + (0.25 if left else -0.25), y + db, f"{bse:.1f}%", ha="left" if left else "right", va="center",
                fontsize=7.6, fontweight="bold", color=INK)
        if round(sev, 1) != round(bse, 1):
            ax.text(sev + (-0.25 if left else 0.25), y + ds, f"{sev:.1f}%", ha="right" if left else "left", va="center",
                    fontsize=7.6, fontweight="bold", color=ORANGE)
        if (k, lab) in ALT:                          # one assumption about recent cohorts, baseline (report §9 sensitivities)
            alt, note = ALT[(k, lab)]
            ax.plot([bse, alt], [y - 0.36, y - 0.36], color=MUTED, lw=0.8, zorder=2)
            ax.scatter([alt], [y - 0.36], s=26, marker="D", facecolor=BG, edgecolor=INK2, linewidth=1, zorder=4)
            ax.text(max(bse, alt) + 0.2, y - 0.36, f"{alt:.1f}%: {note}", va="center", fontsize=6.6, color=INK2)
    ax.set_xlim(0, 11.5); ax.set_ylim(-0.6, len(order) + 0.2)
    row_axis(ax, [o[1] for o in order], "Lifetime loss, % of estimated balance (active book, 2026-09)")
    legend(ax, [("Fed baseline", NAVY, "dot"), ("Fed severely adverse", ORANGE, "dot")], loc="upper right", ncol=2, columnspacing=0.8,
           bbox_to_anchor=(1.0, 1.02))
    panel(ax, letter, title)


# ------------------------------------------------------------------ findings logs (read from the two reports)
SHORT_P = {"F1": "Loan at risk unobserved: property-level prepayment proxy",
           "F2": "New-lien hazard under-predicted out of time by 26–30%",
           "F3": "Sale lock-in coefficient not identified",
           "F4": "Rate proxy attenuated for owners who refinanced before 2022",
           "F5": "Monotone constraint imposes a sale lock-in sign the data do not support",
           "F6": "Unseen-state default over-predicted hold-out sales by 27%",
           "F7": "Exposure-biased 12-month cumulative backtest",
           "F8": "Vendor 'interest rate' is a market-rate lookup, not a recorded rate",
           "F9": "House-price stress response muted",
           "F10": "Three of four rate shocks end outside the training rates"}
SHORT_L = {"L1": "Loan term revised after default: a leak, removed",
           "L2": "Cohort quality confounded with macro sensitivity",
           "L3": "Payment relief broke the unemployment link (7.7× over-prediction)",
           "L4": "Macro path changes 24-month forecasts by <3% (6 of 8 origins; 5 after the lag fix)",
           "L5": "Model choice moves the allowance more than the scenario",
           "L6": "PSI flat while the forecast was 2.3× actual",
           "L7": "Charge-off timing is administrative (write-off batches)",
           "L8": "Severity measured on original amount, not balance",
           "L9": "Rate and collateral fields change across approval eras",
           "L10": "Fits shorter than a full cycle under-predict lifetime losses",
           "L11": "Business-age coding changes across approval eras",
           "L12": "2026Q1–Q3 macro inputs are scenario values, not published data",
           "L13": "Recording lag: September 2026 payoffs not yet in the data",
           "L14": "Era model under-predicts recent cohorts out of time",
           "L15": "Undated charge-offs and date errors: 25 loans, dropped in code",
           "L16": "Scenario changes reverted apart from levels: fixed and re-run"}
SHORT_V = {"V1": "Vendor A AVM 32% below sale prices; stated error uninformative",
           "V2": "Vendor B value anchored to the sale price (leakage)",
           "V3": "Vendor A market value and assessor value are stale, regressive",
           "V4": "No measurable difference in unseen states; splits by period",
           "V5": "Error rose 11.1% to 12.9% while PSI stayed flat",
           "V6": "Conformal intervals under-cover out of time (77.5%)",
           "V7": "Bottom-decile error depends on how tiers are built",
           "V8": "Minority-share gap not robust to tier definition",
           "V9": "Vendor 'interest rate' is a market-rate lookup",
           "V10": "Price-disclosure screen did not bind; result holds without the 12 states"}


FINDING_LOGS = [("credit_loss_cecl_validation.md", "## 12. Findings log", "## 13.", "Credit-loss model"),
                ("prepayment_model_development_and_validation.md", "## 7. Findings log", "## 8.", "Prepayment model"),
                ("avm_champion_challenger_validation.md", "## 8. Findings log", "**Disposition.**", "Home-valuation models")]


def read_findings(doc_name, start, end):
    doc = (ROOT / "docs" / doc_name).read_text()
    log = doc[doc.index(start):doc.index(end)]
    return [(f, sev.strip(), re.sub(r"\*|\(.*?\)", "", st).strip()) for f, sev, st in re.findall(r"^\| ([FLV]\d+) \| ([^|]+) \| ([^|]+) \|", log, flags=re.M)]


def p_matrix(ax):
    """All findings by severity and status."""
    recs = [r for d, s, e, _ in FINDING_LOGS for r in read_findings(d, s, e)]
    sevs, sts = ["High", "Medium", "Low", "Info"], ["Open", "Closed", "Info"]
    cnt = {(a, b): sum(1 for _, s, t in recs if s == a and t == b) for a in sevs for b in sts}
    top = max(cnt.values())
    for j, st in enumerate(sts):
        y = len(sts) - 1 - j
        ax.text(-0.15, y, st, ha="right", va="center", fontsize=8, fontweight="bold", color=INK)
        for i, sv in enumerate(sevs):
            v = cnt[(sv, st)]; base = ORANGE if st == "Open" else (BLUE if st == "Closed" else GREY)
            rbox(ax, i + 0.06, y - 0.4, 0.88, 0.8, mix(PALE, base, 0.15 + 0.85 * v / top) if v else PALE, r=0.1, alpha=1 if v else 0.5)
            ax.text(i + 0.5, y, str(v) if v else "–", ha="center", va="center", fontsize=9, fontweight="bold",
                    color=(ONPILL if v / top > 0.45 else INK) if v else MUTED)
    for i, sv in enumerate(sevs):
        ax.text(i + 0.5, len(sts) - 0.35, sv, ha="center", va="bottom", fontsize=7.5, color=MUTED, fontweight="bold")
    n_open = sum(cnt[(s, "Open")] for s in sevs)
    ax.text(4.35, 1.0, f"{len(recs)} findings\n{n_open} open", ha="left", va="center", fontsize=11, fontweight="bold", color=INK, linespacing=1.3)
    ax.set_xlim(-1.4, 9.4); ax.set_ylim(-0.6, len(sts) + 0.1); ax.axis("off")
    ax.set_title("All findings by severity and status", loc="left", fontweight="bold", color=INK, pad=6)


def figure0():
    """Overview: what each model got right, what it missed, and where they meet."""
    e3 = csv(P3 + "eval_hazard.csv").query("cause == 'co'").set_index(["split", "model"])
    auc, lvl = e3.loc[("out of time 2020-26", "M1 logit"), "auc"], e3.loc[("out of time 2020-26", "M1 logit"), "pred_over_actual"]
    e1 = csv(P1 + "eval_monthly_hazard_final.csv").set_index(["cause", "split", "model"]).pred_over_actual
    a2 = csv(P2 + "accuracy_oot_own_coverage.csv").set_index("estimator")
    cards = [("Write-offs and loss", "SBA 7(a) small-business loans", f"Ranking: AUC {auc:.2f}", f"Write-offs {lvl:.1f}× actual"),
             ("How long loans stay", "Home sales and new mortgages", f"Sales {e1[('sale', 'oot', 'm1')]:.2f}× actual",
              f"New liens {e1[('new_lien', 'oot', 'm1')]:.2f}× actual"),
             ("What collateral is worth", "Vendor and model home values", f"Blend error {100 * a2.loc['C3 blend (C1+C2)', 'MdAPE']:.1f}%",
              "Error rose; PSI quiet")]
    fig = plt.figure(figsize=(7.6, 2.55), dpi=300); ax = fig.add_axes([0.005, 0.005, 0.99, 0.99])
    ax.set_xlim(0, 30); ax.set_ylim(0, 10); ax.axis("off")
    card, edge = mix(BG, PALE, 0.85), mix(PALE, GREY, 0.35)
    for i, (t, sub, held, missed) in enumerate(cards):
        x0 = 0.3 + i * 9.9
        rbox(ax, x0, 3.55, 9.3, 6.3, card, r=0.3, ec=edge, lw=0.6)
        ax.text(x0 + 0.55, 9.0, t, fontsize=9, fontweight="bold", color=INK, va="center")
        ax.text(x0 + 0.55, 8.05, sub, fontsize=7.2, color=MUTED, va="center")
        for k, (lab, txt, col) in enumerate((("Held up", held, BLUE), ("Missed", missed, ORANGE))):
            yy = 6.55 - k * 1.85
            rbox(ax, x0 + 0.45, yy - 0.75, 8.4, 1.5, BG, r=0.25, ec=edge, lw=0.6)
            rbox(ax, x0 + 0.75, yy - 0.45, 2.05, 0.9, col, r=0.22)
            ax.text(x0 + 1.775, yy, lab, fontsize=6.6, color=ONPILL, va="center", ha="center", fontweight="bold")
            ax.text(x0 + 3.15, yy, txt, fontsize=7.8, color=INK, va="center", fontweight="bold")
        ax.annotate("", (x0 + 4.65, 2.3), (x0 + 4.65, 3.45), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, mutation_scale=9,
                    shrinkA=0, shrinkB=0))   # each input feeds the estimate: one straight arrow per card
    rbox(ax, 0.3, 0.2, 29.1, 2.0, NAVY, r=0.3)
    ax.text(15, 1.58, "Lifetime loss estimate", ha="center", va="center", fontsize=9, fontweight="bold", color=ONPILL)
    ax.text(15, 0.78, "Each input held up on one check and missed on another; checks against outcomes showed how large the misses were.",
            ha="center", va="center", fontsize=7.3, color=ONPILL)
    save(fig, "figures/figure0.png")


def p_findings(ax, doc_name, start, end, short, title):
    doc = (ROOT / "docs" / doc_name).read_text()
    log = doc[doc.index(start):doc.index(end)]
    recs = re.findall(r"^\| ([FLV]\d+) \| ([^|]+) \| ([^|]+) \|", log, flags=re.M)
    assert [r[0] for r in recs] == list(short), f"findings log of {doc_name} changed: update the labels"
    sev_c = {"High": ORANGE, "Medium": BLUE, "Low": GREY, "Info": LIGHT}
    n = len(recs)
    for i, (fid, sev, status) in enumerate(recs):
        y = n - 1 - i; sev = sev.strip(); st = re.sub(r"\*|\(.*?\)", "", status).strip()
        rbox(ax, 0.0, y - 0.36, 0.55, 0.72, PALE, r=0.12)
        ax.text(0.275, y, fid, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
        rbox(ax, 0.7, y - 0.34, 1.05, 0.68, sev_c[sev], r=0.12, alpha=0.9)
        ax.text(1.225, y, sev, ha="center", va="center", fontsize=7.5, fontweight="bold", color=ONPILL if sev != "Info" else INK)
        rbox(ax, 1.9, y - 0.34, 1.05, 0.68, {"Open": OSOFT, "Closed": GOOD, "Info": PALE}[st], r=0.12)
        ax.text(2.425, y, st, ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
        ax.text(3.15, y, short[fid], va="center", fontsize=8, color=INK)
    for xx, lab in [(0.275, "ID"), (1.225, "Severity"), (2.425, "Status"), (3.15, "Finding")]:
        ax.text(xx, n - 0.2, lab, ha="center" if xx < 3 else "left", fontsize=7.5, color=MUTED, fontweight="bold")
    ax.set_xlim(-0.05, 9.4); ax.set_ylim(-0.6, n + 0.1); ax.axis("off")
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=6)


# ------------------------------------------------------------------ figures, in README order
def figure1():
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.4), dpi=300, layout="constrained", gridspec_kw=dict(width_ratios=[1.05, 1.0, 0.85]))
    p_bt_overview(ax[0], "A", "Write-offs: 24-month $ forecast ÷ actual")
    p_auc(ax[1], "B", "Loan life: ranking of next-month sales")
    p_avm(ax[2], "C", "Collateral: median valuation error")
    save(fig, "figures/figure1.png")


def figure2():
    fig, ax = plt.subplots(1, 3, figsize=(10.6, 3.4), dpi=300, layout="constrained", gridspec_kw=dict(width_ratios=[0.95, 1.05, 1.0]))
    p_term_field(ax[0], "A", "The term field is revised after default  ·  L1")
    p_leak_auc(ax[1], "B", "What the leak did to the challenger")
    p_vintage(ax[2], "C", "Champion's error by approval year  ·  L2")
    save(fig, "figures/figure2.png")


def figure3():
    fig, ax = plt.subplots(1, 3, figsize=(10.6, 3.5), dpi=300, layout="constrained", gridspec_kw=dict(width_ratios=[1.1, 1.05, 0.9]))
    p_backtest2(ax[0], "A", "24-month backtest, before and after fixes")
    p_pandemic2(ax[1], "B", "Payment relief, 2020–21  ·  L3")
    p_lifetime(ax[2], "C", "Lifetime-to-date backtest")
    save(fig, "figures/figure3.png")


def figure4():
    fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.5), dpi=300, layout="constrained", gridspec_kw=dict(width_ratios=[1.35, 0.9, 0.85]))
    p_allowance2(ax[0], "A", "Lifetime loss by model and scenario  ·  L5")
    p_ur(ax[1], "B", "Fed 2026 unemployment paths")
    p_severity_year(ax[2], "C", "Write-off ÷ original amount, by year")
    save(fig, "figures/figure4.png")


def figure5():
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.4), dpi=300, layout="constrained", gridspec_kw=dict(width_ratios=[1.0, 0.95, 1.0]))
    p_ratio(ax[0], "A", "Calibration by split  ·  F2")
    p_calib(ax[1], "B", "12-month forecast by risk decile")
    p_or(ax[2], "C", "Rate sensitivity across specifications")
    save(fig, "figures/figure5.png")


def figure6():
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.4), dpi=300, layout="constrained", gridspec_kw=dict(width_ratios=[1.15, 1.0, 0.85]))
    p_stress(ax[0], "A", "Rate shocks, 24-month runoff  ·  F10")
    p_hpi(ax[1], "B", "House-price shocks  ·  F9")
    p_ltv(ax[2], "C", "Training data by LTV")
    save(fig, "figures/figure6.png")


def figure7():
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2), dpi=300, layout="constrained", gridspec_kw=dict(width_ratios=[1.2, 1.0]))
    p_nl_month(ax[0], "A", "New-lien shortfall, month by month  ·  F2")
    p_mae(ax[1], "B", "What reduced the error")
    save(fig, "figures/figure7.png")


def figure8():
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.4), dpi=300, layout="constrained", gridspec_kw=dict(width_ratios=[1.25, 0.95, 0.95]))
    p_vendor_q(ax[0], "A", "Vendor values by sale quarter  ·  V1, V2")
    p_geo(ax[1], "B", "Blend vs. vendor, same sales  ·  V4")
    p_drift(ax[2], "C", "Monitoring back-test  ·  V5")
    save(fig, "figures/figure8.png")


def figure9():
    fig, ax = plt.subplots(4, 1, figsize=(7.6, 13.4), dpi=300, layout="constrained",
                           gridspec_kw=dict(height_ratios=[3.6, len(SHORT_L) + 0.7, len(SHORT_P) + 0.7, len(SHORT_V) + 0.7]))
    p_matrix(ax[0]); ax = ax[1:]
    p_findings(ax[0], "credit_loss_cecl_validation.md", "## 12. Findings log", "## 13.", SHORT_L, "Credit-loss model")
    p_findings(ax[1], "prepayment_model_development_and_validation.md", "## 7. Findings log", "## 8.", SHORT_P, "Prepayment model")
    p_findings(ax[2], "avm_champion_challenger_validation.md", "## 8. Findings log", "**Disposition.**", SHORT_V, "Home-valuation models")
    save(fig, "figures/figure9.png")


if __name__ == "__main__":
    style()
    for f in (figure0, figure1, figure2, figure3, figure4, figure5, figure6, figure7, figure8, figure9):
        f()
    if THEME == "light":                          # then the GitHub dark-mode variants
        import os, subprocess
        subprocess.run([sys.executable, __file__], env=dict(os.environ, FIG_THEME="dark"), check=True)

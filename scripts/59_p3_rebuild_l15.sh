#!/bin/zsh
# Credit-loss study check (2026-11-19): the whole pipeline rebuilt with the L15 input checks.
#
# The published tables were built before scripts/40_p3_build.py dropped the 16 undated charge-offs and the 9 loans
# disbursed more than a year before approval (finding L15; LEGACY_FILTERS = True reproduces them). This copies the
# project to a separate folder, rebuilds every credit-loss table there from the public SBA and Fed files with the
# checks on, re-runs the checks of scripts 52-58, and writes the comparison table
# reports/p3/r6_rebuild_l15_comparison.csv (scripts/59_p3_rebuild_comparison.py) back into this project.
set -e
SRC="${0:A:h:h}"; DST="${SRC}_rebuild_l15"
rm -rf "$DST"; mkdir -p "$DST/data_work/p3" "$DST/reports/p3" "$DST/tmp" "$DST/docs"
cp -R "$SRC/scripts" "$SRC/src" "$DST/"; ln -s "$SRC/data_external" "$DST/data_external"
cp "$SRC/docs/credit_loss_plan.md" "$DST/docs/"; cp "$SRC/reports/p3/plan_hash.json" "$DST/reports/p3/"; cp -R "$SRC/reports/p3/before_2026-11-19" "$DST/reports/p3/before_2026-11-19"
cd "$DST"; export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
python3 scripts/40_p3_build.py && python3 scripts/41_p3_panel.py && python3 scripts/42_p3_models.py
for s in 42b_p3_vintage 42c_p3_remediation 42d_p3_descriptives 42e_p3_input_checks 43_p3_backtest 44_p3_cecl 45_p3_remediation2; do python3 scripts/$s.py & done; wait
for s in 46_p3_cecl_r2 48_p3_backtest_v2 50_p3_allowance_sensitivities 51_p3_inference; do python3 scripts/$s.py & done
python3 scripts/47_p3_lifetime_backtest.py & python3 scripts/47_p3_lifetime_backtest.py --no-overlay & wait
for s in 49_p3_monitoring 52_p3_input_and_path_checks 53_p3_backtest_lag_by_origin; do python3 scripts/$s.py & done; wait
for s in 54_p3_allowance_history_and_channels 55_p3_inference_time_blocks 56_p3_severity_book_term_checks 58_p3_backtest_lag_by_origin_r2; do python3 scripts/$s.py & done; wait
cd "$SRC" && python3 scripts/59_p3_rebuild_comparison.py "$DST/reports/p3"

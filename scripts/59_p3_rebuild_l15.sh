#!/bin/zsh
# Credit-loss study check (2026-11-19): the whole pipeline rebuilt with the L15 input checks.
#
# The published tables were built before scripts/40_p3_build.py dropped the 16 undated charge-offs and the 9 loans
# disbursed more than a year before approval (finding L15; LEGACY_FILTERS = True reproduces them). This copies the
# project to a separate folder, rebuilds every credit-loss table there from the public SBA and Fed files with the
# checks on, re-runs the checks of scripts 52-58, and writes the comparison table
# reports/p3/r6_rebuild_l15_comparison.csv (scripts/59_p3_rebuild_comparison.py) back into this project.
set -e
unset MVAL_WORK                                    # never inherit another working folder
par() { local pids=() p; for c in "$@"; do eval "$c" & pids+=($!); done; for p in $pids; do wait $p; done; }   # stop on any failed job
SRC="${0:A:h:h}"; DST="$(mktemp -d "${SRC}_rebuild_l15.XXXXXX")"   # a new folder each run; nothing is deleted
mkdir -p "$DST/data_work/p3" "$DST/reports/p3" "$DST/tmp" "$DST/docs"
cp -R "$SRC/scripts" "$SRC/src" "$DST/"; ln -s "$SRC/data_external" "$DST/data_external"
cp "$SRC/docs/credit_loss_plan.md" "$DST/docs/"; cp "$SRC/reports/p3/plan_hash.json" "$DST/reports/p3/"; cp -R "$SRC/reports/p3/before_2026-11-19" "$DST/reports/p3/before_2026-11-19"
sed -i '' 's/^LEGACY_FILTERS = .*/LEGACY_FILTERS = False      # set by scripts\/59_p3_rebuild_l15.sh/' "$DST/scripts/40_p3_build.py"
grep -q "^LEGACY_FILTERS = False" "$DST/scripts/40_p3_build.py"
cd "$DST"; export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
python3 scripts/40_p3_build.py && python3 scripts/41_p3_panel.py && python3 scripts/42_p3_models.py
par "python3 scripts/42b_p3_vintage.py" "python3 scripts/42c_p3_remediation.py" "python3 scripts/42d_p3_descriptives.py" "python3 scripts/42e_p3_input_checks.py" "python3 scripts/43_p3_backtest.py" "python3 scripts/44_p3_cecl.py" "python3 scripts/45_p3_remediation2.py"
par "python3 scripts/46_p3_cecl_r2.py" "python3 scripts/48_p3_backtest_v2.py" "python3 scripts/50_p3_allowance_sensitivities.py" "python3 scripts/51_p3_inference.py" "python3 scripts/47_p3_lifetime_backtest.py" "python3 scripts/47_p3_lifetime_backtest.py --no-overlay"
par "python3 scripts/49_p3_monitoring.py" "python3 scripts/52_p3_input_and_path_checks.py" "python3 scripts/53_p3_backtest_lag_by_origin.py"
par "python3 scripts/54_p3_allowance_history_and_channels.py" "python3 scripts/55_p3_inference_time_blocks.py" "python3 scripts/56_p3_severity_book_term_checks.py" "python3 scripts/58_p3_backtest_lag_by_origin_r2.py"
cd "$SRC" && python3 scripts/59_p3_rebuild_comparison.py "$DST/reports/p3"

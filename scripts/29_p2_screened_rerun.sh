#!/bin/zsh
# Valuation study check (2026-11-19): the whole pipeline re-run on the states that pass the price-disclosure screen.
#
# scripts/10_p2_sales.py computes each state's share of priced resales with AVG((sa1 >= 30000)::int), which skips
# resales with no price, so every state passes (finding V10). This run copies the project to a separate working folder,
# changes that one line to AVG((COALESCE(sa1, 0) >= 30000)::int) so the share is over all resales (12 states fail),
# and re-runs 10-15, 23, 25 and 26 there. The licensed working tables (data_work/props, the house-price index) are
# linked read-only, so the house-price index is the published one (not refitted without the 12 states). Its result
# tables are copied back as reports/p2/r6_screened_*.
set -e
unset MVAL_WORK                                    # never inherit another working folder
par() { local pids=() p; for c in "$@"; do eval "$c" & pids+=($!); done; for p in $pids; do wait $p; done; }   # stop on any failed job
SRC="${0:A:h:h}"; DST="$(mktemp -d "${SRC}_screened.XXXXXX")"   # a new folder each run; nothing is deleted
mkdir -p "$DST/data_work/p2" "$DST/reports/p2" "$DST/tmp"
cp -R "$SRC/scripts" "$SRC/src" "$SRC/service" "$DST/"
for f in props hpi_state_monthly.parquet hpi_lookup.csv hpi_lookup_xh.csv; do ln -s "$SRC/data_work/$f" "$DST/data_work/$f"; done
sed -i '' 's/avg((sa1 >= 30000)::int) amt_share/avg((COALESCE(sa1, 0) >= 30000)::int) amt_share/' "$DST/scripts/10_p2_sales.py"
cd "$DST"; export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
python3 scripts/10_p2_sales.py && python3 scripts/11_p2_features.py
par "python3 scripts/12_p2_vendor_dd.py" "python3 scripts/13_p2_challenger.py" "python3 scripts/14_p2_attention.py"
python3 scripts/15_p2_eval.py && python3 scripts/23_p2_service_reference.py
par "python3 scripts/25_p2_exante_tiers.py" "python3 scripts/26_p2_robustness.py"
for f in accuracy_oot_own_coverage.csv accuracy_spatial_own_coverage.csv r4_accuracy_spatial_oot.csv r4_monitoring_backtest.csv; do
  cp "reports/p2/$f" "$SRC/reports/p2/r6_screened_$f"; done
cp reports/p2/r4_bootstrap_b1000.json "$SRC/reports/p2/r6_screened_bootstrap_b1000.json"

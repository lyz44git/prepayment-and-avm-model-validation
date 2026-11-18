#!/bin/sh
# Public inputs of the credit-loss study: SBA 7(a) loan files as of 2026-09-30 (U.S. Government Works) and the
# Federal Reserve's 2026 supervisory scenarios. About 760 MB. Nothing is stored in this repository.
set -e
mkdir -p data_external/sba data_external/fed2026
B=https://data.sba.gov/sites/default/files/uploaded_resources
for f in 7a_504_foia_data_dictionary.xlsx FOIA_7a_FY2000_FY2009_asof_260930.csv FOIA_7a_FY2010_FY2019_asof_260930.csv FOIA_7a_FY2020_Present_asof_260930.csv; do
  curl -fsSL "$B/$f" -o "data_external/sba/$f"
done
for f in 2026_Final_Historic_Domestic.csv 2026_Final_Supervisory_Baseline_Domestic.csv 2026_Final_Supervisory_Severely_Adverse_Domestic.csv; do
  curl -fsSL "https://www.federalreserve.gov/supervisionreg/files/$f" -o "data_external/fed2026/$f"
done
(cd data_external/sba && shasum -a 256 -c ../../reports/p3/SHA256SUMS.txt)

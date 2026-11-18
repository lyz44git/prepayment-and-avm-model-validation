#!/bin/sh
# Public macro inputs (FRED): mortgage rates, 10-year Treasury, unemployment, FHFA California index.
# Files are not stored in this repository; this fetches them into data_external/fred/.
set -e
mkdir -p data_external/fred
for id in MORTGAGE30US MORTGAGE15US DGS10 UNRATE CASTHPI; do
  curl -fsSL "https://fred.stlouisfed.org/graph/fredgraph.csv?id=$id" -o "data_external/fred/$id.csv"
done

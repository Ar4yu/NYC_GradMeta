#!/usr/bin/env bash
set -euo pipefail

echo "==> Building NYC mobility daily"
python src/nyc_gradmeta/data/inspect_data.py

# Download Google Trends daily only if we don't already have a processed file.
TRENDS_CSV="data/processed/trends_us_ny_daily.csv"
if [ -f "${TRENDS_CSV}" ]; then
  echo "==> Google Trends daily already exists at ${TRENDS_CSV}, skipping download."
else
  echo "==> Downloading Google Trends daily (pytrends)"
  python src/nyc_gradmeta/data/download_trends_daily.py
fi

echo "==> Building NYC cases/deaths/hospitalizations daily"
python src/nyc_gradmeta/data/build_cases_nyc.py

echo "==> Building master daily (cases + mobility + trends; no OpenTable)"
python src/nyc_gradmeta/data/build_master_daily.py

echo "==> Building OpenTable YoY daily (separate)"
python src/nyc_gradmeta/data/build_opentable_yoy.py

echo "Done. Outputs:"
ls -lh data/processed | sed -n '1,200p'

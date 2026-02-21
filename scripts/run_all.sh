#!/usr/bin/env bash
set -euo pipefail

# End-to-end NYC GradMeta pipeline:
#   1) Build processed public datasets (cases, mobility, trends, OpenTable CSV)
#   2) Build OpenTable private tensor (per-age-patch)
#   3) Build online train/test CSVs for a given ASOF date
#   4) Train and run the NN -> SEIRM -> error-correction adapter forecaster
#
# Usage:
#   ./scripts/run_all.sh 2021-12-31
# or:
#   ASOF=2021-12-31 ./scripts/run_all.sh

ASOF="${1:-${ASOF:-2021-12-31}}"
CFG="configs/nyc.json"
if [ -x ".venv/bin/python" ]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

echo "==> Using ASOF=${ASOF}"

echo "==> Step 1: build processed public datasets"
./scripts/build_data.sh

echo "==> Step 2: build OpenTable private tensor"
"$PYTHON" scripts/build_private_opentable_tensor.py \
  --config "${CFG}" \
  --asof "${ASOF}" \
  --opentable_csv "data/processed/opentable_yoy_daily.csv" \
  --opentable_col "yoy_seated_diner"

echo "==> Step 3: prepare online train/test CSVs"
"$PYTHON" scripts/prepare_online_nyc.py --config "${CFG}" --asof "${ASOF}"

echo "==> Step 4: train + forecast with NN -> SEIRM -> error-correction adapter"
"$PYTHON" -m nyc_gradmeta.models.forecasting_gradmeta_nyc --config "${CFG}" --asof "${ASOF}"

echo "Pipeline completed."

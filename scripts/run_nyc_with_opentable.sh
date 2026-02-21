#!/usr/bin/env bash
set -euo pipefail

# NYC GradMeta pipeline using master public data + OpenTable private tensor.
#
# Usage:
#   ./scripts/run_nyc_with_opentable.sh 2022-10-15
# or:
#   ASOF=2022-10-15 ./scripts/run_nyc_with_opentable.sh

ASOF="${1:-${ASOF:-2022-10-15}}"
CFG="configs/nyc.json"
if [ -x ".venv/bin/python" ]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

echo "==> [MASTER + OPENTABLE] Using ASOF=${ASOF}"

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

echo "==> Step 4: train + forecast with NN -> SEIRM -> error-correction adapter (with private OpenTable)"
"$PYTHON" -m nyc_gradmeta.models.forecasting_gradmeta_nyc --config "${CFG}" --asof "${ASOF}"

echo "[MASTER + OPENTABLE] Pipeline completed."

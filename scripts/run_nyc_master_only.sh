#!/usr/bin/env bash
set -euo pipefail

# Baseline NYC GradMeta pipeline using ONLY master public data
# (no OpenTable private tensor).
#
# Usage:
#   ./scripts/run_nyc_master_only.sh 2022-10-15
# or:
#   ASOF=2022-10-15 ./scripts/run_nyc_master_only.sh

ASOF="${1:-${ASOF:-2022-10-15}}"
CFG="configs/nyc.json"

echo "==> [MASTER ONLY] Using ASOF=${ASOF}"

echo "==> Step 1: build processed public datasets"
./scripts/build_data.sh

echo "==> Step 2: prepare online train/test CSVs"
python scripts/prepare_online_nyc.py --config "${CFG}" --asof "${ASOF}"

echo "==> Step 3: train + forecast with NN -> SEIRM -> error-correction adapter (no private OpenTable)"
python -m nyc_gradmeta.models.forecasting_gradmeta_nyc --config "${CFG}" --asof "${ASOF}" --no_private

echo "[MASTER ONLY] Pipeline completed."


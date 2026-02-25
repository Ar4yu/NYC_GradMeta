#!/usr/bin/env bash
set -euo pipefail

# Baseline NYC GradMeta pipeline using ONLY master public data
# (no OpenTable private tensor).
#
# Usage:
#   ./scripts/run_nyc_master_only.sh 2022-10-15
# or:
#   ASOF=2022-10-15 ./scripts/run_nyc_master_only.sh
#
# Optional env knobs:
#   STAGE=all|gradmeta|adapter|together   (default: all)
#   USE_ADAPTER=1|0                        (default: 0)
#   LONG_TRAIN=1                           (adds --long_train)
#   CLIP_NORM=10                           (used when LONG_TRAIN=1)

ASOF="${1:-${ASOF:-2022-10-15}}"
CFG="configs/nyc.json"
STAGE="${STAGE:-all}"
USE_ADAPTER="${USE_ADAPTER:-0}"
LONG_TRAIN="${LONG_TRAIN:-0}"
CLIP_NORM="${CLIP_NORM:-}"
if [ -x ".venv/bin/python" ]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

echo "==> [MASTER ONLY] Using ASOF=${ASOF}"

echo "==> Step 1: build processed public datasets"
./scripts/build_data.sh

echo "==> Step 2: prepare online train/test CSVs"
"$PYTHON" scripts/prepare_online_nyc.py --config "${CFG}" --asof "${ASOF}"

echo "==> Step 3: staged train + forecast (no private OpenTable)"
TRAIN_ARGS=( -m nyc_gradmeta.models.forecasting_gradmeta_nyc --config "${CFG}" --asof "${ASOF}" --no_private --stage "${STAGE}" )
if [ "${USE_ADAPTER}" = "1" ]; then
  TRAIN_ARGS+=( --use_adapter )
fi
if [ "${LONG_TRAIN}" = "1" ]; then
  TRAIN_ARGS+=( --long_train )
  if [ -n "${CLIP_NORM}" ]; then
    TRAIN_ARGS+=( --clip_norm "${CLIP_NORM}" )
  fi
fi
"$PYTHON" "${TRAIN_ARGS[@]}"

echo "[MASTER ONLY] Pipeline completed."

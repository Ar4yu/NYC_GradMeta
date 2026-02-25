#!/usr/bin/env bash
set -euo pipefail

# NYC GradMeta pipeline using master public data + OpenTable private tensor.
#
# Usage:
#   ./scripts/run_nyc_with_opentable.sh 2022-10-15
# or:
#   ASOF=2022-10-15 ./scripts/run_nyc_with_opentable.sh
#
# Optional env knobs:
#   STAGE=all|gradmeta|adapter|together   (default: all)
#   USE_ADAPTER=1|0                        (default: 1)
#   LONG_TRAIN=1                           (adds --long_train)
#   CLIP_NORM=10                           (used when LONG_TRAIN=1)

ASOF="${1:-${ASOF:-2022-10-15}}"
CFG="configs/nyc.json"
STAGE="${STAGE:-all}"
USE_ADAPTER="${USE_ADAPTER:-1}"
LONG_TRAIN="${LONG_TRAIN:-0}"
CLIP_NORM="${CLIP_NORM:-}"
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

echo "==> Step 4: staged train + forecast (with private OpenTable)"
TRAIN_ARGS=( -m nyc_gradmeta.models.forecasting_gradmeta_nyc --config "${CFG}" --asof "${ASOF}" --stage "${STAGE}" )
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

echo "[MASTER + OPENTABLE] Pipeline completed."

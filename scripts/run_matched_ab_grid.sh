#!/usr/bin/env bash
set -euo pipefail

# Run the fair matched-window A/B grid:
#   A: public_only  with smoothing window 0, 3, 7
#   B: public_opentable with smoothing window 0, 3, 7
#
# Contract:
# - Uses the true observed public/OpenTable overlap window.
# - Reserves the final 28 days for test.
# - Forces both A and B to use the exact same matched dates for each smoothing window.
#
# Usage:
#   ./scripts/run_matched_ab_grid.sh 2022-10-15
# Optional env knobs:
#   STAGE=all|gradmeta|adapter|together
#   USE_ADAPTER=1|0
#   LONG_TRAIN=1|0
#   CLIP_NORM=10
#   PYTHON=/path/to/python

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

COMMON_TRAIN_ARGS=(
  -m nyc_gradmeta.models.forecasting_gradmeta_nyc
  --config "${CFG}"
  --asof "${ASOF}"
  --stage "${STAGE}"
  --matched_window_with_opentable
)
if [ "${USE_ADAPTER}" = "1" ]; then
  COMMON_TRAIN_ARGS+=( --use_adapter )
fi
if [ "${LONG_TRAIN}" = "1" ]; then
  COMMON_TRAIN_ARGS+=( --long_train )
  if [ -n "${CLIP_NORM}" ]; then
    COMMON_TRAIN_ARGS+=( --clip_norm "${CLIP_NORM}" )
  fi
fi

echo "==> Building processed public data"
./scripts/build_data.sh

echo "==> Building matched-window OpenTable private tensor"
"$PYTHON" scripts/build_private_opentable_tensor.py \
  --config "${CFG}" \
  --asof "${ASOF}" \
  --opentable_csv "data/processed/opentable_yoy_daily.csv" \
  --opentable_col "yoy_seated_diner" \
  --matched_window_with_opentable

for SMOOTH_W in 0 3 7; do
  echo "==> Preparing matched public artifacts for smooth_cases_window=${SMOOTH_W}"
  "$PYTHON" scripts/prepare_online_nyc.py \
    --config "${CFG}" \
    --asof "${ASOF}" \
    --smooth_cases_window "${SMOOTH_W}" \
    --matched_window_with_opentable \
    --opentable_csv "data/processed/opentable_yoy_daily.csv" \
    --opentable_col "yoy_seated_diner"

  echo "==> Running A: public_only, w=${SMOOTH_W}"
  "$PYTHON" "${COMMON_TRAIN_ARGS[@]}" \
    --no_private \
    --smooth_cases_window "${SMOOTH_W}"

  VIS_A_ARGS=(
    -m nyc_gradmeta.visualization
    --asof "${ASOF}"
    --config "${CFG}"
    --mode master_only
    --smooth_cases_window "${SMOOTH_W}"
    --matched_window_with_opentable
  )
  if [ "${USE_ADAPTER}" = "1" ]; then
    VIS_A_ARGS+=( --use_adapter )
  fi
  MPLCONFIGDIR="${MPLCONFIGDIR:-.venv/mplconfig}" "$PYTHON" "${VIS_A_ARGS[@]}"

  echo "==> Running B: public_opentable, w=${SMOOTH_W}"
  "$PYTHON" "${COMMON_TRAIN_ARGS[@]}" \
    --smooth_cases_window "${SMOOTH_W}"

  VIS_B_ARGS=(
    -m nyc_gradmeta.visualization
    --asof "${ASOF}"
    --config "${CFG}"
    --mode master_opentable
    --smooth_cases_window "${SMOOTH_W}"
    --matched_window_with_opentable
  )
  if [ "${USE_ADAPTER}" = "1" ]; then
    VIS_B_ARGS+=( --use_adapter )
  fi
  MPLCONFIGDIR="${MPLCONFIGDIR:-.venv/mplconfig}" "$PYTHON" "${VIS_B_ARGS[@]}"
done

echo "==> Final matched split contract"
"$PYTHON" - "${ASOF}" <<'PY'
import json
import sys
from pathlib import Path

asof = sys.argv[1]
base = Path("data/processed/online")
for w in (0, 3, 7):
    path = base / f"split_info_{asof}_matched_ot_w{w}.json"
    if not path.exists():
        continue
    info = json.loads(path.read_text())
    print(
        f"w={w}: window={info['window_start']} -> {info['window_end']}, "
        f"train={info['train_start']} -> {info['train_end']}, "
        f"test={info['test_start']} -> {info['test_end']}, "
        f"window_days={info['window_days']}"
    )
PY

echo "Done. Matched-window A/B artifacts are under outputs/nyc/${ASOF}/."

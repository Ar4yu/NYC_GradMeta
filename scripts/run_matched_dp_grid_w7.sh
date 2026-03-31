#!/usr/bin/env bash
set -euo pipefail

# Run the thesis-facing matched-window DP grid for NYC OpenTable at w=7.
#
# Contract:
# - Resolves the requested ASOF to the true matched public/OpenTable overlap end date.
# - Uses the same matched train/test dates for public_only, non-private OpenTable, and all DP runs.
# - Runs long training by default.
#
# Usage:
#   chmod +x scripts/run_matched_dp_grid_w7.sh
#   ./scripts/run_matched_dp_grid_w7.sh 2022-10-15

REQUESTED_ASOF="${1:-${ASOF:-2022-10-15}}"
CFG="configs/nyc.json"
STAGE="${STAGE:-all}"
USE_ADAPTER="${USE_ADAPTER:-1}"
LONG_TRAIN="${LONG_TRAIN:-1}"
CLIP_NORM="${CLIP_NORM:-10}"
VAL_SPLIT="${VAL_SPLIT:-0}"
DP_DELTA="${DP_DELTA:-1e-4}"
DP_TMAX="${DP_TMAX:-200}"
DP_D="${DP_D:-80000}"
DP_CLIPPING_BOUND_PP="${DP_CLIPPING_BOUND_PP:-100}"
DP_SEED="${DP_SEED:-0}"

if [ -x ".venv/bin/python" ]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

SMOOTH_W=7
EPSILONS=(1 2 4 8 16)

echo "==> Building processed public data"
./scripts/build_data.sh

RUN_ASOF="$("$PYTHON" - "$REQUESTED_ASOF" <<'PY'
import sys
import json
from pathlib import Path
import pandas as pd

requested_asof = pd.to_datetime(sys.argv[1])
cfg = json.load(open("configs/nyc.json", "r", encoding="utf-8"))
master_path = Path(cfg["nyc"]["paths"]["master_daily_csv"])
ot_path = Path("data/processed/opentable_yoy_daily.csv")

master_df = pd.read_csv(master_path)
master_df["date"] = pd.to_datetime(master_df["date"])
public_end = master_df["date"].max()

ot_df = pd.read_csv(ot_path)
ot_df["date"] = pd.to_datetime(ot_df["date"])
observed = ot_df[ot_df["yoy_seated_diner"].notna()]
ot_end = observed["date"].max()
actual = min(public_end, ot_end, requested_asof)
print(actual.strftime("%Y-%m-%d"))
PY
)"

echo "==> Requested ASOF: ${REQUESTED_ASOF}"
echo "==> Resolved matched-window ASOF: ${RUN_ASOF}"

echo "==> Preparing matched public artifacts (w=${SMOOTH_W})"
"$PYTHON" scripts/prepare_online_nyc.py \
  --config "${CFG}" \
  --asof "${RUN_ASOF}" \
  --smooth_cases_window "${SMOOTH_W}" \
  --matched_window_with_opentable \
  --opentable_csv "data/processed/opentable_yoy_daily.csv" \
  --opentable_col "yoy_seated_diner"

COMMON_TRAIN_ARGS=(
  -m nyc_gradmeta.models.forecasting_gradmeta_nyc
  --config "${CFG}"
  --asof "${RUN_ASOF}"
  --stage "${STAGE}"
  --smooth_cases_window "${SMOOTH_W}"
  --matched_window_with_opentable
  --val_split "${VAL_SPLIT}"
)
if [ "${USE_ADAPTER}" = "1" ]; then
  COMMON_TRAIN_ARGS+=( --use_adapter )
fi
if [ "${LONG_TRAIN}" = "1" ]; then
  COMMON_TRAIN_ARGS+=( --long_train --clip_norm "${CLIP_NORM}" )
fi

COMMON_VIS_ARGS=(
  -m nyc_gradmeta.visualization
  --asof "${RUN_ASOF}"
  --config "${CFG}"
  --smooth_cases_window "${SMOOTH_W}"
  --matched_window_with_opentable
)
if [ "${USE_ADAPTER}" = "1" ]; then
  COMMON_VIS_ARGS+=( --use_adapter )
fi

echo "==> Building matched non-private OpenTable tensor"
"$PYTHON" scripts/build_private_opentable_tensor.py \
  --config "${CFG}" \
  --asof "${RUN_ASOF}" \
  --opentable_csv "data/processed/opentable_yoy_daily.csv" \
  --opentable_col "yoy_seated_diner" \
  --matched_window_with_opentable \
  --clipping_bound_pp "${DP_CLIPPING_BOUND_PP}"

echo "==> Running baseline A: public_only_adapter_w7_matched_ot"
"$PYTHON" "${COMMON_TRAIN_ARGS[@]}" --no_private
MPLCONFIGDIR="${MPLCONFIGDIR:-.venv/mplconfig}" "$PYTHON" "${COMMON_VIS_ARGS[@]}" --mode master_only

echo "==> Running baseline B: public_opentable_adapter_w7_matched_ot"
"$PYTHON" "${COMMON_TRAIN_ARGS[@]}"
MPLCONFIGDIR="${MPLCONFIGDIR:-.venv/mplconfig}" "$PYTHON" "${COMMON_VIS_ARGS[@]}" --mode master_opentable

for PRIVACY_MODE in event restaurant; do
  for EPS in "${EPSILONS[@]}"; do
    echo "==> Running DP OpenTable: mode=${PRIVACY_MODE}, epsilon=${EPS}"
    "$PYTHON" scripts/build_private_opentable_tensor.py \
      --config "${CFG}" \
      --asof "${RUN_ASOF}" \
      --opentable_csv "data/processed/opentable_yoy_daily.csv" \
      --opentable_col "yoy_seated_diner" \
      --matched_window_with_opentable \
      --privacy_mode "${PRIVACY_MODE}" \
      --mechanism gaussian \
      --epsilon "${EPS}" \
      --delta "${DP_DELTA}" \
      --tmax "${DP_TMAX}" \
      --denominator_d "${DP_D}" \
      --clipping_bound_pp "${DP_CLIPPING_BOUND_PP}" \
      --dp_seed "${DP_SEED}"

    "$PYTHON" "${COMMON_TRAIN_ARGS[@]}" \
      --privacy_mode "${PRIVACY_MODE}" \
      --mechanism gaussian \
      --epsilon "${EPS}"

    MPLCONFIGDIR="${MPLCONFIGDIR:-.venv/mplconfig}" "$PYTHON" "${COMMON_VIS_ARGS[@]}" \
      --mode master_opentable \
      --privacy_mode "${PRIVACY_MODE}" \
      --mechanism gaussian \
      --epsilon "${EPS}"
  done
done

echo "==> Building DP comparison summary and plots"
"$PYTHON" scripts/plot_matched_dp_summary.py --asof "${RUN_ASOF}" --config "${CFG}"

echo "Done."
echo "Resolved output directory: outputs/nyc/${RUN_ASOF}/"
echo "DP summary CSV: outputs/nyc/${RUN_ASOF}/metrics_summary_dp_w7.csv"

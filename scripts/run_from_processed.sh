#!/usr/bin/env bash
# Run train + save + visualize using already processed data.
# Requires: data/processed/online/train_<ASOF>.csv, test_<ASOF>.csv,
#           data/processed/private/opentable_private_lap_<ASOF>.pt (unless --no_private),
#           data/processed/population_us.csv, data/processed/contact_matrix_us.csv
#
# Usage (activate .venv first, or set PYTHON):
#   . .venv/bin/activate && ./scripts/run_from_processed.sh 2022-10-15
#   ./scripts/run_from_processed.sh 2022-10-15 --no_private
#   ./scripts/run_from_processed.sh 2022-10-15 --epochs 10   # quick test
# Optional env knobs:
#   STAGE=all|gradmeta|adapter|together   (default: all)
#   USE_ADAPTER=1|0                        (default: 1)
#   LONG_TRAIN=1                           (adds --long_train)
#   CLIP_NORM=10                           (used when LONG_TRAIN=1)

set -euo pipefail

if [ -x ".venv/bin/python" ]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi
ASOF="${1:?Usage: $0 <ASOF> [--no_private] [--epochs N]}"
shift || true
STAGE="${STAGE:-all}"
USE_ADAPTER="${USE_ADAPTER:-1}"
LONG_TRAIN="${LONG_TRAIN:-0}"
CLIP_NORM="${CLIP_NORM:-}"
NO_PRIVATE=""
EXTRA_ARGS=()
MODE="master_opentable"
for arg in "$@"; do
  if [ "$arg" = "--no_private" ]; then
    NO_PRIVATE="--no_private"
    MODE="master_only"
  else
    EXTRA_ARGS+=("$arg")
  fi
done

CFG="configs/nyc.json"

echo "==> Training + saving model (ASOF=${ASOF}, mode=${MODE})"
TRAIN_ARGS=( -m nyc_gradmeta.models.forecasting_gradmeta_nyc --config "${CFG}" --asof "${ASOF}" --stage "${STAGE}" )
if [ -n "${NO_PRIVATE}" ]; then
  TRAIN_ARGS+=( "${NO_PRIVATE}" )
fi
if [ "${USE_ADAPTER}" = "1" ]; then
  TRAIN_ARGS+=( --use_adapter )
fi
if [ "${LONG_TRAIN}" = "1" ]; then
  TRAIN_ARGS+=( --long_train )
  if [ -n "${CLIP_NORM}" ]; then
    TRAIN_ARGS+=( --clip_norm "${CLIP_NORM}" )
  fi
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  TRAIN_ARGS+=( "${EXTRA_ARGS[@]}" )
fi
"$PYTHON" "${TRAIN_ARGS[@]}"

echo "==> Saving visualization"
MPLCONFIGDIR="${MPLCONFIGDIR:-.venv/mplconfig}" "$PYTHON" -m nyc_gradmeta.visualization \
  --asof "${ASOF}" \
  --config "${CFG}" \
  --mode "${MODE}"

echo "Done. Forecast and models: outputs/nyc/${ASOF}/"
echo "Plot: results/nyc_forecast_${MODE}_${ASOF}.png"

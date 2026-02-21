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

set -euo pipefail

if [ -x ".venv/bin/python" ]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi
ASOF="${1:?Usage: $0 <ASOF> [--no_private] [--epochs N]}"
shift || true
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
"$PYTHON" -m nyc_gradmeta.models.forecasting_gradmeta_nyc \
  --config "${CFG}" \
  --asof "${ASOF}" \
  ${NO_PRIVATE} \
  "${EXTRA_ARGS[@]}"

echo "==> Saving visualization"
MPLCONFIGDIR="${MPLCONFIGDIR:-.venv/mplconfig}" "$PYTHON" -m nyc_gradmeta.visualization \
  --asof "${ASOF}" \
  --config "${CFG}" \
  --mode "${MODE}"

echo "Done. Forecast and models: outputs/nyc/${ASOF}/"
echo "Plot: results/nyc_forecast_${MODE}_${ASOF}.png"

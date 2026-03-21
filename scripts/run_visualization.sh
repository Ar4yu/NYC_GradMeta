#!/usr/bin/env bash
set -euo pipefail

# Generate forecast-vs-truth plot for a given ASOF and mode.
# Run after training (e.g. after run_nyc_master_only.sh or run_nyc_with_opentable.sh).
#
# Usage:
#   ./scripts/run_visualization.sh 2022-10-15 master_only
#   ./scripts/run_visualization.sh 2022-10-15 master_opentable

ASOF="${1:-2022-10-15}"
MODE="${2:-master_opentable}"
SMOOTH_CASES_WINDOW="${3:-0}"
WINDOW_DAYS="${WINDOW_DAYS:-170}"
CFG="configs/nyc.json"
if [ -x ".venv/bin/python" ]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

echo "==> Saving visualization ASOF=${ASOF} mode=${MODE} smooth_cases_window=${SMOOTH_CASES_WINDOW}"
"$PYTHON" -m nyc_gradmeta.visualization \
  --asof "${ASOF}" \
  --config "${CFG}" \
  --mode "${MODE}" \
  --smooth_cases_window "${SMOOTH_CASES_WINDOW}" \
  --window_days "${WINDOW_DAYS}"
echo "Done. Check outputs/nyc/${ASOF}/forecast_vs_truth_${MODE}_w${SMOOTH_CASES_WINDOW}.png"

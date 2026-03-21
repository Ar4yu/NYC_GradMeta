#!/usr/bin/env bash
set -euo pipefail

ASOF="${1:-2022-10-15}"
CFG="${CFG:-configs/nyc.json}"
WINDOW_DAYS="${WINDOW_DAYS:-170}"

if [ -x ".venv/bin/python" ]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

echo "==> Preparing smoothed public datasets for ASOF=${ASOF} (window_days=${WINDOW_DAYS})"
for SMOOTH_W in 3 7; do
  "$PYTHON" scripts/prepare_online_nyc.py \
    --config "${CFG}" \
    --asof "${ASOF}" \
    --window_days "${WINDOW_DAYS}" \
    --smooth_cases_window "${SMOOTH_W}"
done

echo "==> Building OpenTable private tensor once"
"$PYTHON" scripts/build_private_opentable_tensor.py \
  --config "${CFG}" \
  --asof "${ASOF}" \
  --opentable_csv "data/processed/opentable_yoy_daily.csv" \
  --opentable_col "yoy_seated_diner"

echo "==> Running 1-epoch smoke forecasts"
for SMOOTH_W in 3 7; do
  echo "-- smooth_cases_window=${SMOOTH_W}"
  "$PYTHON" -m nyc_gradmeta.models.forecasting_gradmeta_nyc \
    --config "${CFG}" \
    --asof "${ASOF}" \
    --window_days "${WINDOW_DAYS}" \
    --smooth_cases_window "${SMOOTH_W}" \
    --epochs 1 \
    --self_check
done

echo "==> Smoke test complete. Outputs are under outputs/nyc/${ASOF}/"

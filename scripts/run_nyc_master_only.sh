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

POSITIONAL=()
SKIP_PREP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-prep)
      SKIP_PREP=1
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"
ASOF="${1:-${ASOF:-2022-10-15}}"
EXTRA_ARGS=()
if [ "${#POSITIONAL[@]}" -gt 1 ]; then
  EXTRA_ARGS=("${POSITIONAL[@]:1}")
fi
CFG="configs/nyc.json"
STAGE="${STAGE:-all}"
USE_ADAPTER="${USE_ADAPTER:-0}"
LONG_TRAIN="${LONG_TRAIN:-0}"
CLIP_NORM="${CLIP_NORM:-}"
SMOOTH_CASES_WINDOW="${SMOOTH_CASES_WINDOW:-0}"
WINDOW_DAYS="${WINDOW_DAYS:-170}"
MATCHED_WINDOW_WITH_OPENTABLE="${MATCHED_WINDOW_WITH_OPENTABLE:-0}"
if [ -x ".venv/bin/python" ]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

echo "==> [MASTER ONLY] Using ASOF=${ASOF}"

if [[ "$SKIP_PREP" -eq 0 ]]; then
  echo "==> Step 1: build processed public datasets"
  ./scripts/build_data.sh
  echo "==> Step 2: prepare online train/test CSVs"
  if [ "${MATCHED_WINDOW_WITH_OPENTABLE}" = "1" ]; then
    "$PYTHON" scripts/prepare_online_nyc.py \
      --config "${CFG}" \
      --asof "${ASOF}" \
      --window_days "${WINDOW_DAYS}" \
      --smooth_cases_window "${SMOOTH_CASES_WINDOW}" \
      --matched_window_with_opentable \
      --opentable_csv "data/processed/opentable_yoy_daily.csv" \
      --opentable_col "yoy_seated_diner"
  else
    "$PYTHON" scripts/prepare_online_nyc.py \
      --config "${CFG}" \
      --asof "${ASOF}" \
      --window_days "${WINDOW_DAYS}" \
      --smooth_cases_window "${SMOOTH_CASES_WINDOW}"
  fi
else
  echo "==> Skipping prep steps (build_data/prepare_online)"
fi

echo "==> Step 3: staged train + forecast (no private OpenTable)"
TRAIN_ARGS=(
  -m nyc_gradmeta.models.forecasting_gradmeta_nyc
  --config "${CFG}"
  --asof "${ASOF}"
  --no_private
  --stage "${STAGE}"
  --window_days "${WINDOW_DAYS}"
  --smooth_cases_window "${SMOOTH_CASES_WINDOW}"
)
if [ "${MATCHED_WINDOW_WITH_OPENTABLE}" = "1" ]; then
  TRAIN_ARGS+=( --matched_window_with_opentable )
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
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
"$PYTHON" "${TRAIN_ARGS[@]}"

echo "[MASTER ONLY] Pipeline completed."

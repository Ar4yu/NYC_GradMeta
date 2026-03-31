#!/usr/bin/env bash
set -euo pipefail

# NYC GradMeta pipeline using master public data + OpenTable private tensor.
#
# Usage:
#   ./scripts/run_nyc_with_opentable.sh 2022-10-15 [--skip-prep]
# or:
#   ASOF=2022-10-15 ./scripts/run_nyc_with_opentable.sh [--skip-prep]
#
# Optional env knobs:
#   STAGE=all|gradmeta|adapter|together   (default: all)
#   USE_ADAPTER=1|0                        (default: 1)
#   LONG_TRAIN=1                           (adds --long_train)
#   CLIP_NORM=10                           (used when LONG_TRAIN=1)
#   --skip-prep (or SKIP_PREP=1)            skip rebuild steps 1-3

SKIP_PREP=0
POSITIONAL=()
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
USE_ADAPTER="${USE_ADAPTER:-1}"
LONG_TRAIN="${LONG_TRAIN:-0}"
CLIP_NORM="${CLIP_NORM:-}"
SMOOTH_CASES_WINDOW="${SMOOTH_CASES_WINDOW:-0}"
WINDOW_DAYS="${WINDOW_DAYS:-170}"
MATCHED_WINDOW_WITH_OPENTABLE="${MATCHED_WINDOW_WITH_OPENTABLE:-0}"
DP_PRIVACY_MODE="${DP_PRIVACY_MODE:-none}"
DP_MECHANISM="${DP_MECHANISM:-gaussian}"
DP_EPSILON="${DP_EPSILON:-}"
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

echo "==> [MASTER + OPENTABLE] Using ASOF=${ASOF}"

if [[ "$SKIP_PREP" -eq 0 ]]; then
  echo "==> Step 1: build processed public datasets"
  ./scripts/build_data.sh

  echo "==> Step 2: build OpenTable private tensor"
  OT_ARGS=(
    scripts/build_private_opentable_tensor.py
    --config "${CFG}"
    --asof "${ASOF}"
    --opentable_csv "data/processed/opentable_yoy_daily.csv"
    --opentable_col "yoy_seated_diner"
  )
  if [ "${MATCHED_WINDOW_WITH_OPENTABLE}" = "1" ]; then
    OT_ARGS+=( --matched_window_with_opentable )
  fi
  if [ "${DP_PRIVACY_MODE}" != "none" ]; then
    OT_ARGS+=( --privacy_mode "${DP_PRIVACY_MODE}" --mechanism "${DP_MECHANISM}" --epsilon "${DP_EPSILON}" --delta "${DP_DELTA}" --tmax "${DP_TMAX}" --denominator_d "${DP_D}" --clipping_bound_pp "${DP_CLIPPING_BOUND_PP}" --dp_seed "${DP_SEED}" )
  fi
  "$PYTHON" "${OT_ARGS[@]}"

  echo "==> Step 3: prepare online train/test CSVs"
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
  echo "==> Skipping prep steps (build_data/build_private/prepare_online)"
fi

echo "==> Step 4: staged train + forecast (with private OpenTable)"
TRAIN_ARGS=(
  -m nyc_gradmeta.models.forecasting_gradmeta_nyc
  --config "${CFG}"
  --asof "${ASOF}"
  --stage "${STAGE}"
  --window_days "${WINDOW_DAYS}"
  --smooth_cases_window "${SMOOTH_CASES_WINDOW}"
)
if [ "${MATCHED_WINDOW_WITH_OPENTABLE}" = "1" ]; then
  TRAIN_ARGS+=( --matched_window_with_opentable )
fi
if [ "${DP_PRIVACY_MODE}" != "none" ]; then
  TRAIN_ARGS+=( --privacy_mode "${DP_PRIVACY_MODE}" --mechanism "${DP_MECHANISM}" --epsilon "${DP_EPSILON}" )
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

echo "[MASTER + OPENTABLE] Pipeline completed."

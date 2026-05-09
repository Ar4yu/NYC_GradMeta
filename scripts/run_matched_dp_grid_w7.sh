#!/usr/bin/env bash
set -euo pipefail

# Run the thesis-facing matched-window DP grid for NYC OpenTable at w=7.
#
# Contract:
# - Resolves the requested ASOF to the true matched public/OpenTable overlap end date.
# - Uses the same matched train/test dates for public_only, non-private OpenTable, and all DP runs.
# - Runs long training by default.
# - Legacy single-seed invocation still writes to outputs/nyc/<ASOF>/.
# - Passing --seeds and/or --experiment-tag writes an isolated run group:
#   outputs/nyc/<ASOF>/run_groups/<experiment-tag>/seed_<N>/ and aggregate/.
#
# Usage:
#   chmod +x scripts/run_matched_dp_grid_w7.sh
#   ./scripts/run_matched_dp_grid_w7.sh 2022-10-15
#   ./scripts/run_matched_dp_grid_w7.sh --long-train --seeds 0,1,2,3,4 --experiment-tag dp_multiseed_$(date +%Y%m%d_%H%M%S)

usage() {
  cat <<'EOF'
Usage: run_matched_dp_grid_w7.sh [ASOF|--asof YYYY-MM-DD] [--long-train|--no-long-train]
                                 [--seeds 0,1,2,3,4] [--experiment-tag TAG]
                                 [--clip_norm N|--no-clip-norm]
                                 [--epochs_gradmeta N] [--epochs_adapter N] [--epochs_together N]
                                 [--val_split F] [--patience N] [--baseline-only] [--allow-existing]

Defaults:
  ASOF: $ASOF env or 2022-10-15
  legacy mode: one seed, DP_SEED env or 0, outputs/nyc/<resolved-ASOF>/
  run-group mode: enabled by --seeds or --experiment-tag; seed list defaults to 0,1,2,3,4

Run-group outputs:
  outputs/nyc/<resolved-ASOF>/run_groups/<TAG>/seed_<N>/
  outputs/nyc/<resolved-ASOF>/run_groups/<TAG>/aggregate/
  thesis_visualizations/*_<TAG>_aggregate.{csv,pdf}
  thesis_visualizations_png/*_<TAG>_aggregate.png
EOF
}

REQUESTED_ASOF="${ASOF:-2022-10-15}"
CFG="configs/nyc.json"
STAGE="${STAGE:-all}"
USE_ADAPTER="${USE_ADAPTER:-1}"
LONG_TRAIN="${LONG_TRAIN:-1}"
LONG_TRAIN_EXPLICIT=0
CLIP_NORM="${CLIP_NORM:-10}"
VAL_SPLIT="${VAL_SPLIT:-0}"
PATIENCE="${PATIENCE:-50}"
DP_DELTA="${DP_DELTA:-1e-4}"
DP_TMAX="${DP_TMAX:-200}"
DP_D="${DP_D:-80000}"
DP_CLIPPING_BOUND_PP="${DP_CLIPPING_BOUND_PP:-100}"
DP_SEED="${DP_SEED:-0}"
EPOCHS_GRADMETA="${EPOCHS_GRADMETA:-}"
EPOCHS_ADAPTER="${EPOCHS_ADAPTER:-}"
EPOCHS_TOGETHER="${EPOCHS_TOGETHER:-}"
BASELINE_ONLY="${BASELINE_ONLY:-0}"
SEEDS_RAW=""
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
ALLOW_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asof)
      REQUESTED_ASOF="$2"
      shift 2
      ;;
    --config)
      CFG="$2"
      shift 2
      ;;
    --long-train|--long_train)
      LONG_TRAIN=1
      LONG_TRAIN_EXPLICIT=1
      shift
      ;;
    --no-long-train|--no_long_train)
      LONG_TRAIN=0
      LONG_TRAIN_EXPLICIT=1
      shift
      ;;
    --epochs_gradmeta)
      EPOCHS_GRADMETA="$2"
      shift 2
      ;;
    --clip_norm)
      CLIP_NORM="$2"
      shift 2
      ;;
    --no-clip-norm|--no_clip_norm)
      CLIP_NORM=""
      shift
      ;;
    --epochs_adapter)
      EPOCHS_ADAPTER="$2"
      shift 2
      ;;
    --epochs_together)
      EPOCHS_TOGETHER="$2"
      shift 2
      ;;
    --val_split)
      VAL_SPLIT="$2"
      shift 2
      ;;
    --patience)
      PATIENCE="$2"
      shift 2
      ;;
    --baseline-only|--baseline_only)
      BASELINE_ONLY=1
      shift
      ;;
    --seeds)
      SEEDS_RAW="$2"
      shift 2
      ;;
    --experiment-tag|--experiment_tag)
      EXPERIMENT_TAG="$2"
      shift 2
      ;;
    --allow-existing)
      ALLOW_EXISTING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
    *)
      REQUESTED_ASOF="$1"
      shift
      ;;
  esac
done

if [ -x ".venv/bin/python" ]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

SMOOTH_W=7
EPSILONS=(1 2 4 8 16)

RUN_GROUP_MODE=0
if [[ -n "${SEEDS_RAW}" || -n "${EXPERIMENT_TAG}" ]]; then
  RUN_GROUP_MODE=1
fi

parse_seed_list() {
  local raw="$1"
  raw="${raw//,/ }"
  SEEDS=()
  for seed in $raw; do
    SEEDS+=("$seed")
  done
}

if [[ "${RUN_GROUP_MODE}" -eq 1 ]]; then
  if [[ -z "${SEEDS_RAW}" ]]; then
    SEEDS_RAW="0,1,2,3,4"
  fi
  if [[ -z "${EXPERIMENT_TAG}" ]]; then
    EXPERIMENT_TAG="dp_multiseed_$(date +%Y%m%d_%H%M%S)"
  fi
  parse_seed_list "${SEEDS_RAW}"
else
  SEEDS=("${DP_SEED}")
fi

if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "No seeds provided." >&2
  exit 1
fi

echo "==> Building processed public data"
./scripts/build_data.sh

RUN_ASOF="$("$PYTHON" - "$REQUESTED_ASOF" "$CFG" <<'PY'
import sys
import json
from pathlib import Path
import pandas as pd

requested_asof = pd.to_datetime(sys.argv[1])
cfg = json.load(open(sys.argv[2], "r", encoding="utf-8"))
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

RUN_GROUP_DIR=""
if [[ "${RUN_GROUP_MODE}" -eq 1 ]]; then
  RUN_GROUP_DIR="outputs/nyc/${RUN_ASOF}/run_groups/${EXPERIMENT_TAG}"
  if [[ -e "${RUN_GROUP_DIR}" && "${ALLOW_EXISTING}" -ne 1 ]]; then
    echo "Refusing to reuse existing run group: ${RUN_GROUP_DIR}" >&2
    echo "Choose a new --experiment-tag or pass --allow-existing if you intentionally want to append/replace files there." >&2
    exit 1
  fi
  mkdir -p "${RUN_GROUP_DIR}"
  printf "%s\n" "${SEEDS[@]}" > "${RUN_GROUP_DIR}/seeds.txt"
  echo "==> Run-group output directory: ${RUN_GROUP_DIR}"
fi

echo "==> Preparing matched public artifacts (w=${SMOOTH_W})"
"$PYTHON" scripts/prepare_online_nyc.py \
  --config "${CFG}" \
  --asof "${RUN_ASOF}" \
  --smooth_cases_window "${SMOOTH_W}" \
  --matched_window_with_opentable \
  --opentable_csv "data/processed/opentable_yoy_daily.csv" \
  --opentable_col "yoy_seated_diner"

make_seed_config() {
  local seed="$1"
  local seed_dir="$2"
  local cfg_out="${seed_dir}/config_seed_${seed}.json"
  mkdir -p "${seed_dir}/private"
  "$PYTHON" - "${CFG}" "${cfg_out}" "${seed_dir}/private" <<'PY'
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
private_dir = Path(sys.argv[3])
cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
cfg["nyc"]["paths"]["private_dir"] = private_dir.as_posix()
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY
  echo "${cfg_out}"
}

run_one_seed() {
  local seed="$1"
  local current_cfg="${CFG}"
  local output_dir=""
  local output_args=()
  local train_seed_args=()
  local explicit_epoch_override=0

  if [[ "${RUN_GROUP_MODE}" -eq 1 ]]; then
    output_dir="${RUN_GROUP_DIR}/seed_${seed}"
    if [[ -e "${output_dir}" && "${ALLOW_EXISTING}" -ne 1 ]]; then
      echo "Refusing to reuse existing seed output directory: ${output_dir}" >&2
      exit 1
    fi
    mkdir -p "${output_dir}"
    current_cfg="$(make_seed_config "${seed}" "${output_dir}")"
    output_args=( --output_dir "${output_dir}" )
    train_seed_args=( --seed "${seed}" )
  fi

  local common_train_args=(
    -m nyc_gradmeta.models.forecasting_gradmeta_nyc
    --config "${current_cfg}"
    --asof "${RUN_ASOF}"
    --stage "${STAGE}"
    --smooth_cases_window "${SMOOTH_W}"
    --matched_window_with_opentable
    --val_split "${VAL_SPLIT}"
    --patience "${PATIENCE}"
  )
  if [[ -n "${EPOCHS_GRADMETA}" ]]; then
    common_train_args+=( --epochs_gradmeta "${EPOCHS_GRADMETA}" )
    explicit_epoch_override=1
  fi
  if [[ -n "${EPOCHS_ADAPTER}" ]]; then
    common_train_args+=( --epochs_adapter "${EPOCHS_ADAPTER}" )
    explicit_epoch_override=1
  fi
  if [[ -n "${EPOCHS_TOGETHER}" ]]; then
    common_train_args+=( --epochs_together "${EPOCHS_TOGETHER}" )
    explicit_epoch_override=1
  fi
  if [ "${USE_ADAPTER}" = "1" ]; then
    common_train_args+=( --use_adapter )
  fi
  if [[ -n "${CLIP_NORM}" && "${CLIP_NORM}" != "0" ]]; then
    common_train_args+=( --clip_norm "${CLIP_NORM}" )
  fi
  if [ "${LONG_TRAIN}" = "1" ] && { [ "${explicit_epoch_override}" -eq 0 ] || [ "${LONG_TRAIN_EXPLICIT}" -eq 1 ]; }; then
    common_train_args+=( --long_train )
  elif [ "${LONG_TRAIN}" = "0" ]; then
    common_train_args+=( --no_long_train )
  fi

  local common_vis_args=(
    -m nyc_gradmeta.visualization
    --asof "${RUN_ASOF}"
    --config "${current_cfg}"
    --smooth_cases_window "${SMOOTH_W}"
    --matched_window_with_opentable
  )
  if [ "${USE_ADAPTER}" = "1" ]; then
    common_vis_args+=( --use_adapter )
  fi

  echo "==> Building matched non-private OpenTable tensor (seed=${seed})"
  "$PYTHON" scripts/build_private_opentable_tensor.py \
    --config "${current_cfg}" \
    --asof "${RUN_ASOF}" \
    --opentable_csv "data/processed/opentable_yoy_daily.csv" \
    --opentable_col "yoy_seated_diner" \
    --matched_window_with_opentable \
    --clipping_bound_pp "${DP_CLIPPING_BOUND_PP}"

  echo "==> Running baseline A: public_only_adapter_w7_matched_ot (seed=${seed})"
  "$PYTHON" "${common_train_args[@]}" "${train_seed_args[@]}" "${output_args[@]}" --no_private
  MPLCONFIGDIR="${MPLCONFIGDIR:-.venv/mplconfig}" "$PYTHON" "${common_vis_args[@]}" "${output_args[@]}" --mode master_only

  echo "==> Running baseline B: public_opentable_adapter_w7_matched_ot (seed=${seed})"
  "$PYTHON" "${common_train_args[@]}" "${train_seed_args[@]}" "${output_args[@]}"
  MPLCONFIGDIR="${MPLCONFIGDIR:-.venv/mplconfig}" "$PYTHON" "${common_vis_args[@]}" "${output_args[@]}" --mode master_opentable

  if [[ "${BASELINE_ONLY}" -ne 1 ]]; then
    for privacy_mode in event restaurant; do
      for eps in "${EPSILONS[@]}"; do
        echo "==> Running DP OpenTable: seed=${seed}, mode=${privacy_mode}, epsilon=${eps}"
        "$PYTHON" scripts/build_private_opentable_tensor.py \
          --config "${current_cfg}" \
          --asof "${RUN_ASOF}" \
          --opentable_csv "data/processed/opentable_yoy_daily.csv" \
          --opentable_col "yoy_seated_diner" \
          --matched_window_with_opentable \
          --privacy_mode "${privacy_mode}" \
          --mechanism gaussian \
          --epsilon "${eps}" \
          --delta "${DP_DELTA}" \
          --tmax "${DP_TMAX}" \
          --denominator_d "${DP_D}" \
          --clipping_bound_pp "${DP_CLIPPING_BOUND_PP}" \
          --dp_seed "${seed}"

        "$PYTHON" "${common_train_args[@]}" "${train_seed_args[@]}" "${output_args[@]}" \
          --privacy_mode "${privacy_mode}" \
          --mechanism gaussian \
          --epsilon "${eps}"

        MPLCONFIGDIR="${MPLCONFIGDIR:-.venv/mplconfig}" "$PYTHON" "${common_vis_args[@]}" "${output_args[@]}" \
          --mode master_opentable \
          --privacy_mode "${privacy_mode}" \
          --mechanism gaussian \
          --epsilon "${eps}"
      done
    done
  fi

  echo "==> Building per-seed DP comparison summary and plots (seed=${seed})"
  "$PYTHON" scripts/plot_matched_dp_summary.py --asof "${RUN_ASOF}" --config "${current_cfg}" "${output_args[@]}"
}

for seed in "${SEEDS[@]}"; do
  run_one_seed "${seed}"
done

if [[ "${RUN_GROUP_MODE}" -eq 1 ]]; then
  SEEDS_JOINED="$(IFS=,; echo "${SEEDS[*]}")"
  echo "==> Aggregating matched summaries across seeds: ${SEEDS_JOINED}"
  "$PYTHON" scripts/plot_matched_dp_summary.py \
    --asof "${RUN_ASOF}" \
    --config "${CFG}" \
    --run-group-dir "${RUN_GROUP_DIR}" \
    --seeds "${SEEDS_JOINED}" \
    --experiment-tag "${EXPERIMENT_TAG}"
fi

echo "Done."
if [[ "${RUN_GROUP_MODE}" -eq 1 ]]; then
  echo "Resolved run-group directory: ${RUN_GROUP_DIR}/"
  echo "Per-seed outputs: ${RUN_GROUP_DIR}/seed_<seed>/"
  echo "Aggregate summary CSV: ${RUN_GROUP_DIR}/aggregate/metrics_summary_dp_w7_aggregate.csv"
  echo "Aggregate baseline CSV: ${RUN_GROUP_DIR}/aggregate/metrics_summary_dp_w7_baselines_aggregate.csv"
  echo "Paired baseline CSV: ${RUN_GROUP_DIR}/aggregate/baseline_paired_differences_w7_matched_ot.csv"
else
  echo "Resolved output directory: outputs/nyc/${RUN_ASOF}/"
  echo "DP summary CSV: outputs/nyc/${RUN_ASOF}/metrics_summary_dp_w7.csv"
fi

#!/usr/bin/env bash
set -euo pipefail

# Unified training runner for NYC GradMeta.
# - Supports public-only and public+OpenTable modes.
# - Reuses existing train/test/private artifacts by default (no clutter).
# - Can rebuild artifacts with --force-prep or skip prep with --skip-prep.
#
# Usage examples:
#   ./scripts/train_nyc.sh --asof 2022-10-15 --mode public_only --skip-prep
#   ./scripts/train_nyc.sh --asof 2022-10-15 --mode opentable      --skip-prep
#   ./scripts/train_nyc.sh --asof 2022-10-15 --mode opentable      --force-prep
#
# Env knobs:
#   EPOCHS=50               # backward-compatible base epoch override
#   EPOCHS_GRADMETA=200     # stage-specific epochs
#   EPOCHS_ADAPTER=100
#   EPOCHS_TOGETHER=200
#   STAGE=all               # gradmeta|adapter|together|all
#   USE_ADAPTER=1           # add --use_adapter
#   ADAPTER_LOSS=mse        # mse|rmse
#   PYTHON=/custom/python
#   OPENTABLE_CSV=...  # override private input CSV (default: data/processed/opentable_yoy_daily.csv)
#   OPENTABLE_COL=...  # override column used in private builder (default: yoy_seated_diner)

usage() {
  cat <<'EOF'
Usage: train_nyc.sh [--asof YYYY-MM-DD] [--mode public_only|opentable] [--force-prep|--skip-prep] [--long]
                    [--stage gradmeta|adapter|together|all]
                    [--epochs-gradmeta N] [--epochs-adapter N] [--epochs-together N]
                    [--adapter-loss mse|rmse] [--use-adapter]
Defaults:
  --asof defaults to $ASOF env or 2022-10-15
  --mode public_only
  --stage all
  prep: auto (run only if files missing). Use --force-prep to rebuild, --skip-prep to require existing files.
  --long enables long-training regimen (num_epochs_long or 5x default) and passes clip norm if set.
EOF
}

ASOF="${ASOF:-}"
MODE="${MODE:-public_only}"
LONG=0
CLIP_NORM="${CLIP_NORM:-}"
USE_ADAPTER=0
STAGE="${STAGE:-all}"
ADAPTER_LOSS="${ADAPTER_LOSS:-mse}"
EPOCHS_GRADMETA="${EPOCHS_GRADMETA:-}"
EPOCHS_ADAPTER="${EPOCHS_ADAPTER:-}"
EPOCHS_TOGETHER="${EPOCHS_TOGETHER:-}"
FORCE_PREP=0
SKIP_PREP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asof) ASOF="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --force-prep) FORCE_PREP=1; shift ;;
    --skip-prep) SKIP_PREP=1; shift ;;
    --long) LONG=1; shift ;;
    --use-adapter|--use_adapter) USE_ADAPTER=1; shift ;;
    --stage) STAGE="$2"; shift 2 ;;
    --adapter-loss|--adapter_loss) ADAPTER_LOSS="$2"; shift 2 ;;
    --epochs-gradmeta|--epochs_gradmeta) EPOCHS_GRADMETA="$2"; shift 2 ;;
    --epochs-adapter|--epochs_adapter) EPOCHS_ADAPTER="$2"; shift 2 ;;
    --epochs-together|--epochs_together) EPOCHS_TOGETHER="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

ASOF="${ASOF:-2022-10-15}"

if [ -x ".venv/bin/python" ]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python not found at $PYTHON. Activate your venv or set PYTHON=/path/to/python" >&2
  exit 1
fi

CFG="configs/nyc.json"
OPENTABLE_CSV="${OPENTABLE_CSV:-data/processed/opentable_yoy_daily.csv}"
OPENTABLE_COL="${OPENTABLE_COL:-yoy_seated_diner}"

# Read paths from config (keeps script aligned if paths move); compatible with macOS bash 3.2
IFS=$'\n' read -r ONLINE_DIR PRIVATE_DIR <<EOF
$("$PYTHON" - <<'PY'
import json
paths = json.load(open("configs/nyc.json"))["nyc"]["paths"]
print(paths["online_dir"])
print(paths["private_dir"])
PY
)
EOF
ONLINE_DIR="${ONLINE_DIR:-data/processed/online}"
PRIVATE_DIR="${PRIVATE_DIR:-data/processed/private}"

mode_norm="$(echo "$MODE" | tr '[:upper:]' '[:lower:]')"
case "$mode_norm" in
  public|public_only|master|master_only)
    RUN_MODE="public_only"
    PRIVATE_FLAG="--no_private"
    ;;
  opentable|with_opentable|public_opentable)
    RUN_MODE="public_opentable"
    PRIVATE_FLAG=""
    ;;
  *)
    echo "Invalid --mode '${MODE}'. Use public_only or opentable." >&2
    exit 1
    ;;
esac

case "$STAGE" in
  gradmeta|adapter|together|all) ;;
  *)
    echo "Invalid --stage '${STAGE}'. Use gradmeta|adapter|together|all." >&2
    exit 1
    ;;
esac

case "$ADAPTER_LOSS" in
  mse|rmse) ;;
  *)
    echo "Invalid --adapter-loss '${ADAPTER_LOSS}'. Use mse|rmse." >&2
    exit 1
    ;;
esac

train_csv="${ONLINE_DIR}/train_${ASOF}.csv"
test_csv="${ONLINE_DIR}/test_${ASOF}.csv"
private_pt="${PRIVATE_DIR}/opentable_private_lap_${ASOF}.pt"

need_public_prep=0
need_private_prep=0

if [[ $SKIP_PREP -eq 1 ]]; then
  if [[ ! -f "$train_csv" || ! -f "$test_csv" ]]; then
    echo "[skip-prep] Missing ${train_csv} or ${test_csv}. Remove --skip-prep or run prepare_online_nyc.py first." >&2
    exit 1
  fi
  if [[ "$RUN_MODE" == "public_opentable" && ! -f "$private_pt" ]]; then
    echo "[skip-prep] Missing ${private_pt}. Remove --skip-prep or run build_private_opentable_tensor.py first." >&2
    exit 1
  fi
else
  if [[ $FORCE_PREP -eq 1 || ! -f "$train_csv" || ! -f "$test_csv" ]]; then
    need_public_prep=1
  fi
  if [[ "$RUN_MODE" == "public_opentable" && ($FORCE_PREP -eq 1 || ! -f "$private_pt") ]]; then
    need_private_prep=1
  fi
fi

if [[ $need_public_prep -eq 1 ]]; then
  echo "==> Building public datasets (train/test) for ASOF=${ASOF}"
  ./scripts/build_data.sh
  "$PYTHON" scripts/prepare_online_nyc.py --config "$CFG" --asof "$ASOF"
else
  echo "==> Reusing existing public datasets: $train_csv, $test_csv"
fi

if [[ "$RUN_MODE" == "public_opentable" ]]; then
  if [[ $need_private_prep -eq 1 ]]; then
    echo "==> Building OpenTable private tensor for ASOF=${ASOF}"
    "$PYTHON" scripts/build_private_opentable_tensor.py \
      --config "$CFG" \
      --asof "$ASOF" \
      --opentable_csv "$OPENTABLE_CSV" \
      --opentable_col "$OPENTABLE_COL"
  else
    echo "==> Reusing existing private tensor: $private_pt"
  fi
fi

echo "==> Training (${RUN_MODE}) ASOF=${ASOF}"
echo "    ONLINE_DIR=$ONLINE_DIR"
echo "    PRIVATE_DIR=$PRIVATE_DIR"
TRAIN_ARGS=( -m nyc_gradmeta.models.forecasting_gradmeta_nyc --config "$CFG" --asof "$ASOF" $PRIVATE_FLAG )
TRAIN_ARGS+=( --stage "$STAGE" --adapter_loss "$ADAPTER_LOSS" )
if [[ -n "${EPOCHS:-}" ]]; then
  TRAIN_ARGS+=( --epochs "$EPOCHS" )
fi
if [[ -n "$EPOCHS_GRADMETA" ]]; then
  TRAIN_ARGS+=( --epochs_gradmeta "$EPOCHS_GRADMETA" )
fi
if [[ -n "$EPOCHS_ADAPTER" ]]; then
  TRAIN_ARGS+=( --epochs_adapter "$EPOCHS_ADAPTER" )
fi
if [[ -n "$EPOCHS_TOGETHER" ]]; then
  TRAIN_ARGS+=( --epochs_together "$EPOCHS_TOGETHER" )
fi
if [[ $LONG -eq 1 ]]; then
  TRAIN_ARGS+=( --long_train )
  if [[ -n "$CLIP_NORM" ]]; then
    TRAIN_ARGS+=( --clip_norm "$CLIP_NORM" )
  fi
fi
if [[ $USE_ADAPTER -eq 1 ]]; then
  TRAIN_ARGS+=( --use_adapter )
fi

"$PYTHON" "${TRAIN_ARGS[@]}"

ADAPTER_SUFFIX=""
if [[ $USE_ADAPTER -eq 1 ]]; then
  ADAPTER_SUFFIX="_adapter"
fi
echo "==> Done. Outputs in outputs/nyc/${ASOF}/ (run_tag=${RUN_MODE}${ADAPTER_SUFFIX})."

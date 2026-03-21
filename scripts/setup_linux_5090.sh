#!/usr/bin/env bash
set -euo pipefail

# Linux setup helper for NVIDIA Blackwell / RTX 5090 lab machines.
# Uses a repo-local venv and installs an official CUDA 12.8 PyTorch build,
# which is the safest default for sm_120 / compute capability 12.0 GPUs.

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-5090}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

echo "==> Creating Linux GPU environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing PyTorch with CUDA 12.8 wheels"
python -m pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"

echo "==> Installing repo dependencies"
python -m pip install -e .

echo "==> Running GPU verification"
python scripts/check_gpu_env.py

cat <<'EOF'

Environment ready.

Activate it with:
  source .venv-5090/bin/activate

Run a training command, for example:
  CUDA_VISIBLE_DEVICES=0 ./scripts/run_nyc_with_opentable.sh 2022-10-15 --skip-prep --val_split 0
EOF

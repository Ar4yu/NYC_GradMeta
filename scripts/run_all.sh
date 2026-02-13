#!/usr/bin/env bash
set -euo pipefail

echo "Running NYC GradMeta pipeline (skeleton)..."

python -m nyc_gradmeta.data.build_dataset --config configs/nyc.yaml
python -m nyc_gradmeta.train --config configs/nyc.yaml
python -m nyc_gradmeta.eval --config configs/nyc.yaml

echo "Done."

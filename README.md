# NYC GradMeta (Thesis Runbook)

This repo ports the professor's GradMeta/GradABM forecasting skeleton to NYC with:

- 16 patches (`num_patch=16`, age-stratified)
- weekly piecewise-constant epi parameters (`param_t = params_epi_weekly[t // 7]`)
- daily SEIRM stepping
- 28-day horizon (`days_head=28`)
- optional private OpenTable encoder input

## 1) Quick setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
chmod +x scripts/*.sh
```

If your shell does not have `python`, all scripts now auto-use `.venv/bin/python` when available.

## 2) Data section (what is used, where it lives)

### Core processed inputs

- `data/processed/nyc_master_daily.csv`
  - main public table, daily contiguous time index
  - includes NYC cases/deaths/hospitalizations and public covariates
  - `prepare_online_nyc.py` builds model-ready online files from this
- `data/processed/opentable_yoy_daily.csv`
  - OpenTable citywide daily private signal
  - used to build the private tensor
- `data/processed/contact_matrix_us.csv`
  - 16x16 age contact matrix (used as migration/contact coupling in simulator)
- `data/processed/population_nyc_age16_2020.csv`
  - NYC age-stratified population vector, length 16
- `configs/nyc.json`
  - source of `num_patch`, `num_pub_features`, horizon, learning rate, and file paths

### Model-ready artifacts produced by scripts

- `data/processed/online/train_<ASOF>.csv`
- `data/processed/online/test_<ASOF>.csv`
  - columns: `cases` + exactly `num_pub_features` public features
  - no target leakage into `X` (`cases` is dropped from model input)
  - test split is the final 28 days up to `ASOF`

- `data/processed/private/opentable_private_lap_<ASOF>.pt`
  - tensor shape `[16, T]`
  - citywide OpenTable replicated across patches with population-share weighting
  - forecasting code loads then unsqueezes to `[16, T, 1]`

### Data build commands (ASOF = 2022-10-15)

Build public online train/test:

```bash
.venv/bin/python scripts/prepare_online_nyc.py \
  --asof 2022-10-15 \
  --config configs/nyc.json
```

Build 1-year rolling window with chronological 80/20 split:

```bash
.venv/bin/python scripts/prepare_online_nyc.py \
  --asof 2022-10-15 \
  --config configs/nyc.json \
  --window_days 365 \
  --split_mode ratio \
  --train_ratio 0.8
```

This also writes:
- `data/processed/online/split_info_<ASOF>.json`
- `data/processed/online/public_feature_map_<ASOF>.csv`

Build private OpenTable tensor:

```bash
.venv/bin/python scripts/build_private_opentable_tensor.py \
  --asof 2022-10-15 \
  --opentable_csv data/processed/opentable_yoy_daily.csv \
  --opentable_col yoy_seated_diner \
  --config configs/nyc.json
```

## 3) Training + forecasting commands

### Quick one-liners (reuse existing train/test; no clutter)

- Public-only:  
  ```bash
  ./scripts/train_nyc.sh --asof 2022-10-15 --mode public_only --skip-prep
  ```
- Public + OpenTable:  
  ```bash
  ./scripts/train_nyc.sh --asof 2022-10-15 --mode opentable --skip-prep
  ```

Rebuild artifacts if needed by swapping `--skip-prep` for `--force-prep`.  
Customize epochs with `EPOCHS=50 ./scripts/train_nyc.sh ...`.  
Enable adapter with `USE_ADAPTER=1 ./scripts/train_nyc.sh ...`.
Declutter outputs with `--minimal_outputs` (keeps only run-tag forecasts, metrics, and fit CSV).
Long-training + gradient clipping (more epochs, clip if set):  
```bash
CLIP_NORM=10 ./scripts/train_nyc.sh --asof 2022-10-15 --mode public_only --long --skip-prep
CLIP_NORM=10 ./scripts/train_nyc.sh --asof 2022-10-15 --mode opentable  --long --skip-prep
```

### A) Public-only baseline (no private OpenTable)

Full run (300 epochs from config):
```bash
.venv/bin/python -m nyc_gradmeta.models.forecasting_gradmeta_nyc \
  --asof 2022-10-15 \
  --config configs/nyc.json \
  --no_private
```

### B) Public + OpenTable baseline

Full run (300 epochs from config):
```bash
.venv/bin/python -m nyc_gradmeta.models.forecasting_gradmeta_nyc \
  --asof 2022-10-15 \
  --config configs/nyc.json
```

### C) Fast smoke test

```bash
.venv/bin/python -m nyc_gradmeta.models.forecasting_gradmeta_nyc \
  --asof 2022-10-15 \
  --config configs/nyc.json \
  --epochs 1 \
  --self_check \
  --no_private
```

### D) 1-year data, 80/20 split, full public-only run (long run)

```bash
# 1) Create 1-year, 80/20 online split
.venv/bin/python scripts/prepare_online_nyc.py \
  --asof 2022-10-15 \
  --config configs/nyc.json \
  --window_days 365 \
  --split_mode ratio \
  --train_ratio 0.8

# 2) Train public-only baseline (full epochs)
.venv/bin/python -m nyc_gradmeta.models.forecasting_gradmeta_nyc \
  --asof 2022-10-15 \
  --config configs/nyc.json \
  --no_private \
  --epochs 300
```

Optional overnight/background run:

```bash
mkdir -p logs
nohup .venv/bin/python -m nyc_gradmeta.models.forecasting_gradmeta_nyc \
  --asof 2022-10-15 \
  --config configs/nyc.json \
  --no_private \
  --epochs 300 > logs/public_only_2022-10-15.log 2>&1 &
```

## 4) Where results are saved

All outputs go to:
- `outputs/nyc/2022-10-15/`

Per run, the model now saves:

- model checkpoints:
  - `param_model.pt`
  - `error_adapter.pt` (only if `--use_adapter`)
- forecasts (N = length of test set):
  - default: `forecast_<N>d.npy/.csv` plus run-tagged copies `forecast_<N>d_<run_tag>.npy/.csv`
  - legacy 28-day aliases also written when N=28
  - add `--minimal_outputs` to keep only run-tagged forecasts and skip legacy duplicates
- fit + metrics:
  - `fit_train_test_<run_tag>.csv`
  - `fit_train_test_<run_tag>.png` (omitted when `--minimal_outputs`)
  - `metrics_<run_tag>.json`
  - `metrics_summary.csv` (append/update table for easy comparison across runs)

`metrics_summary.csv` is the main thesis comparison file (RMSE/MAE/MAPE per run mode).

## 5) Visualization commands

### Full train+test fit visualization (recommended)

Generated automatically by training:
- `outputs/nyc/<ASOF>/fit_train_test_<run_tag>.png`

This includes:
- predicted curve over full series
- true curve over full series
- vertical line marking train/test split

### Existing test-window-only visualization

```bash
./scripts/run_visualization.sh 2022-10-15 master_only
./scripts/run_visualization.sh 2022-10-15 master_opentable
```

Writes:
- `results/nyc_forecast_master_only_2022-10-15.png`
- `results/nyc_forecast_master_opentable_2022-10-15.png`

## 6) Single-command pipelines

Public-only:
```bash
./scripts/run_nyc_master_only.sh 2022-10-15
```

Public + OpenTable:
```bash
./scripts/run_nyc_with_opentable.sh 2022-10-15
```

All-in-one:
```bash
./scripts/run_all.sh 2022-10-15
```

From already processed data only:
```bash
./scripts/run_from_processed.sh 2022-10-15 --no_private
./scripts/run_from_processed.sh 2022-10-15
```

## 7) Alignment status vs professor outline

Implemented and aligned:
- weekly params, daily simulator, 28-day horizon, 16 patches
- two-encoder calibration network + SEIRM(Beta) simulator
- no target leakage from `cases` into public `X`
- OpenTable private tensor path and shape contract
- robust matrix/population loading and seed control

Current deliberate simplifications:
- public covariates are now configurable:
  - `all_public` (default): all numeric non-target columns from `nyc_master_daily.csv`
  - `mobility_index`: compressed single mobility feature
  - `trend`: single Google Trends feature
- adapter is optional (`--use_adapter`) and off by default

## 8) Thesis experiment recommendation

Run at least these two modes with same ASOF/seed/epochs:

1. `public_only`
2. `public_opentable`

Then compare rows in:
- `outputs/nyc/<ASOF>/metrics_summary.csv`

Primary decision criterion:
- lower `test_rmse` / `test_mae` in `public_opentable` vs `public_only`.

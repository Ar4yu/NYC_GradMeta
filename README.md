# NYC GradMeta (Thesis Runbook)

This repo mirrors the professor's Bogotá GradMeta/GradABM stack on NYC data. The pipeline:
- encodes private (OpenTable) + public sequences via the two-encoder calibration net
- decodes weekly epi parameters + seeds + beta matrix using Bogotá-style sigmoid → min/max scaling
- runs the Metapopulation SEIRM(Beta) simulator + optional adapter correction with a Bogotá-inspired 3-stage regimen (GradMeta → Adapter → Together)
- ships utilities/scripts for rebuilding processed data, training in stages, and logging reproducible outputs

<details>
<summary>**1) Getting started (keeps README compact)**</summary>

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
chmod +x scripts/*.sh
```

- Every script prefers `.venv/bin/python` when present; override with `PYTHON=/path/to/python` if needed.
- Use `pip install -e .` once per machine to register the package imports.

- **Linux GPU / Conda option**: if you’re running on NVIDIA lab machines, create a CUDA-enabled `conda` env instead:

```bash
conda create -n nyc-gradmeta python=3.10
conda activate nyc-gradmeta
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .
chmod +x scripts/*.sh
```

Inside that environment PyTorch will see the GPU automatically (`torch.cuda.is_available()` returns `True`). Set `CUDA_VISIBLE_DEVICES=0` or similar before running if you need to pin a specific card. All scripts still use the same CLI; just activate the conda env before invoking them.

</details>

<details>
<summary>**1a) After clone / copying between machines**</summary>

Make sure the processed-data tree exists and is populated before you rely on training runs. After `git clone`, run:

```bash
mkdir -p data/processed data/processed/online data/processed/private
./scripts/build_data.sh
.venv/bin/python scripts/build_private_opentable_tensor.py \
  --asof 2022-10-15 \
  --config configs/nyc.json \
  --opentable_csv data/processed/opentable_yoy_daily.csv \
  --opentable_col yoy_seated_diner
```

This recreates `data/processed/nyc_master_daily.csv`, the online `train/test` CSVs, and the OpenTable `[16,T]` tensor, so you can `git pull` on another machine and immediately run `./scripts/train_nyc.sh`. These directories remain in place even if the repo is copied or checked out elsewhere.

</details>

<details>
<summary>**2) Data flows & preprocessing (documented for thesis reproducibility)**</summary>

### Core inputs
- `data/processed/nyc_master_daily.csv`: merged cases/deaths/hosp + mobility + trends, ready for `prepare_online_nyc.py`.
- `data/processed/opentable_yoy_daily.csv`: private OpenTable YoY signal used to build the `[16, T]` tensor.
- `data/processed/contact_matrix_us.csv` and `data/processed/population_nyc_age16_2020.csv`: simulator contact + population constants.
- `configs/nyc.json`: central configuration (patch count, `num_pub_features`, epoch defaults, file paths, biasing ranges, `param_ranges`).

### Generated artifacts per ASOF
- `data/processed/online/train_<ASOF>.csv` / `test_<ASOF>.csv`: final public covariates + target (`cases` column removed from features). `SeqDataset` loads them directly.
- `data/processed/online/public_feature_map_<ASOF>.csv`: column order reference for your thesis methods section.
- `data/processed/online/split_info_<ASOF>.json`: reproducible split metadata (train/test lengths + window mode).
- `data/processed/private/opentable_private_lap_<ASOF>.pt`: `[16, T]` tensor; `align_private_tensor` handles right alignment + normalization.

### External source data for the new NYC baseline
- NYC City daily case/hospitalization data (NYC OpenData COVID Daily Counts). [https://data.cityofnewyork.us/Health/COVID-19-Daily-Counts-of-Cases-Hospitalizations-an/rc75-m7u3/about_data](https://data.cityofnewyork.us/Health/COVID-19-Daily-Counts-of-Cases-Hospitalizations-an/rc75-m7u3/about_data)
- OpenTable reservation dataset (Kaggle). [https://www.kaggle.com/datasets/pizacd/opentable-reservation-data](https://www.kaggle.com/datasets/pizacd/opentable-reservation-data)
- Google RID mobility report (US/NYC). [https://www.google.com/covid19/mobility/](https://www.google.com/covid19/mobility/)
- Google Trends “Covid 19” NYC view (Jan 2020–Jan 2022). [https://trends.google.com/explore?geo=US-NY&q=Covid%252019&date=2020-01-01%202022-01-01](https://trends.google.com/explore?geo=US-NY&q=Covid%252019&date=2020-01-01%202022-01-01)
- Contact matrix (NYC/US) – document which source you copy into `data/processed/contact_matrix_us.csv` when updating the pipeline; if rebuilt, note the provenance in `README.md`.

Collect each dataset in `data/processed` before running the pipeline, keep notes on the source URLs + collection date, and describe any additional preprocessing (e.g., resampling or aggregation) in this README so Professor Nguyen can reproduce the Feb 9 result. Mention whether the OpenTable window or Google Trends window limits the train/test horizon, and keep the contact matrix up to date with the current US configuration.

### Build commands (run before staging, example ASOF=2022-10-15)
```bash
.venv/bin/python scripts/prepare_online_nyc.py --asof 2022-10-15 --config configs/nyc.json
.venv/bin/python scripts/build_private_opentable_tensor.py \
  --asof 2022-10-15 \
  --opentable_csv data/processed/opentable_yoy_daily.csv \
  --opentable_col yoy_seated_diner \
  --config configs/nyc.json
```

The scripts also write:
- `data/processed/online/split_info_<ASOF>.json`
- `data/processed/online/public_feature_map_<ASOF>.csv`
- Private tensor metadata (shape, date range) printed to console.

### Numeric parsing safeguard (important)
- NYC raw case files may contain quoted thousands separators (for example `"1,034"`).  
- The pipeline now strips commas and coercively parses counts in:
  - `src/nyc_gradmeta/data/build_cases_nyc.py`
  - `src/nyc_gradmeta/data/build_master_daily.py`
  - `scripts/prepare_online_nyc.py`
- Any remaining non-numeric tokens are reported and filled as `0` with a warning to avoid silent corruption.

### Feature lagging & temporal context
- `public_feature_map` shows the mapping used by `SeqDataset`: `pub_0=probable_cases`, `pub_1=hospitalizations`, `pub_2=deaths`, `pub_3=mob_retail`, `pub_4=mob_grocery`, `pub_5=mob_parks`, `pub_6=mob_transit`, `pub_7=mob_work`, `pub_8=mob_residential`, `pub_9=trend_covid_topic`.
- The calibration network consumes the most recent `train_days` worth of these public covariates plus the aligned private tensor; `series_to_supervised` builds lagged public features (`n_in=4`, `n_out=1`) so the encoder sees the last four days before decoding weekly parameters.
- Private data is right-aligned to match the train+test window, padded to `[num_patch, train+test, 1]`, and normalized (`private_norm=zscore_time` by default), ensuring the encoder sees consistent historical context.
- When you run staged training, each stage reuses these sequences: Stage 1 builds epi parameters from the historical window, Stage 2 trains the adapter/residual on the same window, and Stage 3 fine-tunes both with the same temporal context. This keeps the pipeline anchored in realistic lagged data and lets the adaptor correct any remaining errors.

</details>

<details>
<summary>**3) Training + forecasting (foolproof commands)**</summary>

### Shell helpers for any experience level
- `scripts/train_nyc.sh`: universal trainer. Flags:
  - `--mode public_only|opentable`
  - `--stage gradmeta|adapter|together|all`
  - `--epochs_gradmeta`, `--epochs_adapter`, `--epochs_together`
  - `--long_train`, `--clip_norm`
  - `--adapter-loss mse|rmse`, `--use_adapter`
  - `--skip-prep`, `--force-prep`, `--minimal_outputs`
- Wrapper scripts call it with sensible defaults and ensure data is built:
  - `scripts/run_nyc_master_only.sh ASOF`
  - `scripts/run_nyc_with_opentable.sh ASOF` (also rebuilds private tensor)
  - `scripts/run_all.sh ASOF` (full pipeline from raw CSVs)
  - `scripts/run_from_processed.sh ASOF [--no_private] [--epochs N]`

### How staged training runs
- Stage 1 `gradmeta`: trains `CalibNNTwoEncoderThreeOutputs` alone against simulator RMSE; the best `param_model.pt` checkpoint is saved.
- Stage 2 `adapter`: freezes that model, trains `ErrorCorrectionAdapter` on `y_train_s - base_preds` (loss type controlled by `--adapter-loss`), and saves `error_adapter.pt`.
- Stage 3 `together`: unfreezes both modules, blends simulator fit with adapter residual loss using an annealing factor, and updates both checkpoints when the combined loss improves.
- `STAGE=all` runs the phases sequentially; short-circuit with `--stage adapter` or `--stage together`. `--long_train` multiplies each stage’s epochs (via active `epochs_*` or the config) and enables gradient clipping (`--clip_norm`).
- See `src/nyc_gradmeta/models/forecasting_gradmeta_nyc.py` for the exact loops and `scripts/train_nyc.sh` for CLI wiring.

### Copy/paste ready commands
```bash
./scripts/run_nyc_master_only.sh 2022-10-15
USE_ADAPTER=1 ./scripts/run_nyc_with_opentable.sh 2022-10-15
LONG_TRAIN=1 CLIP_NORM=10 STAGE=all USE_ADAPTER=1 ./scripts/run_nyc_with_opentable.sh 2022-10-15
```

### Reload full pipeline after preprocessing fixes
```bash
./scripts/build_data.sh
.venv/bin/python scripts/build_private_opentable_tensor.py --config configs/nyc.json --asof 2022-10-15 --opentable_csv data/processed/opentable_yoy_daily.csv --opentable_col yoy_seated_diner
.venv/bin/python scripts/prepare_online_nyc.py --config configs/nyc.json --asof 2022-10-15
LONG_TRAIN=1 CLIP_NORM=10 STAGE=all USE_ADAPTER=1 ./scripts/run_from_processed.sh 2022-10-15
```

### From processed data with stages
```bash
LONG_TRAIN=1 CLIP_NORM=10 STAGE=all USE_ADAPTER=1 ./scripts/run_from_processed.sh 2022-10-15
STAGE=gradmeta ./scripts/run_from_processed.sh 2022-10-15 --no_private
STAGE=adapter ./scripts/run_from_processed.sh 2022-10-15 --epochs 50
```

### For thesis experiments
- Set `STAGE=gradmeta` / `adapter` / `together` to isolate each phase.
- Use `--long_train` + `CLIP_NORM` for the extended regimen used in your recent runs.
- Use `--minimal_outputs` to keep only `run_tag`-specific artifacts when writing to limited storage.

</details>

<details>
<summary>**4) Outputs & artifacts (where to look for results)**</summary>

Runs write into `outputs/nyc/<ASOF>/` (matching `run_tag`). Files:
- `param_model.pt`, `error_adapter.pt` (if adapter stage ran).
- `forecast_<N>d(.csv/.npy)` *and* `forecast_<N>d_<run_tag>.csv/.npy` (plus `forecast_28d` aliases when `N=28`).
- `fit_train_test_<run_tag>.csv` *and* `.png` (unless `--minimal_outputs`).
- `metrics_<run_tag>.json` and `metrics_summary.csv` (appended each run; use for thesis tables).`metrics_summary.csv` is the canonical comparison table with RMSE/MAE/MAPE/seed info.

Logs print stage progress and warnings if predictions exceed 1e7 (blow-up guard). Always verify the PNG at `outputs/nyc/<ASOF>/fit_train_test_<run_tag>.png` before writing your thesis figures.

</details>

<details>
<summary>**5) Bugs encountered + fixes applied (record for your thesis appendix)**summary>

- **Bogotá-style in-network scaling**: epi params, seed vector, and beta matrix now scale inside `CalibNNTwoEncoderThreeOutputs` via sigmoid + configurable min/max buffers (`nyc.param_min/max`, `nyc.seed_min/max`, `nyc.beta_min/max`). No more external `[0,1]` clamping forcing wrong scales.
- **Seed semantics**: `MetapopulationSEIRMBeta` now honors `nyc.seed_mode` (`fraction` vs `count`), clamps to `[0, num_agents]`, and avoids tensor vs scalar `clamp` by separating `min`/`max` operations for compatibility.
- **Private tensor alignment**: `align_private_tensor` now right-aligns to match train/test windows and offers normalization modes `none`, `zscore_time`, `l2_time` (default `zscore_time`). Stats are logged when `--self_check` is on.
- **Staged training**: added three loops (GradMeta → Adapter → Together), stage-specific checkpoints, adapter residual loss options (`mse` or `rmse`), freezing nil weights, long-train multipliers, blow-up guard, and stage-specific metrics written to JSON.
- **Constraint logic**: `forward_simulator` now clamps epi params to configurable min/max rather than `[0,1]`, ensuring consistent magnitudes after in-network scaling.
- **Scripts & documentation**: new CLI flags (`STAGE`, `ADAPTER_LOSS`, `LONG_TRAIN`, etc.), README reorganized around staged workflow, and helper scripts updated to pass the new flags automatically.

</details>

<details>
<summary>**6) Thesis recommendations & next steps**</summary>

1. Run `public_only` and `public_opentable` with identical seeds/epochs, compare `metrics_summary.csv`, and cite the RMSE/MAE gap.
2. Keep the `fit_train_test_<run_tag>.png` from `outputs/nyc/<ASOF>/` for plots; it shows train/test splits plus predictions/ground truth.
3. Save the JSON metrics (`metrics_<run_tag>.json`) per run and include the `best_loss_gradmeta/adapter/together` fields for transparency.
4. If you need smaller runs, set `--stage gradmeta` and `--epochs_gradmeta 50` for a demo without exploding the pipeline.

</details>

<details>
<summary>**7) Data inventory (tables + scripts)**</summary>

| Dataset | Rows | Columns | Description | Generated by |
| --- | --- | --- | --- | --- |
| `data/processed/nyc_master_daily.csv` | 960 | 13 | Combined cases, hospitalizations, deaths, mobility indexes, scope trends, and `total_cases`. | `scripts/build_data.sh` |
| `data/processed/opentable_yoy_daily.csv` | 170 | 2 | OpenTable YoY diners (date + `yoy_seated_diner`). | `scripts/build_private_opentable_tensor.py` (preprocessing steps) |
| `data/processed/online/train_<ASOF>.csv` | ~932 | 11 | Train window: `cases` + `pub_0`…`pub_9`. | `scripts/prepare_online_nyc.py` |
| `data/processed/online/test_<ASOF>.csv` | 28 | 11 | Test window, same schema as train. | `scripts/prepare_online_nyc.py` |
| `data/processed/private/opentable_private_lap_<ASOF>.pt` | 16 × T |  | Per-patch OpenTable tensor, zero-padded + normalized (`align_private_tensor`). | `scripts/build_private_opentable_tensor.py` |

- `scripts/prepare_online_nyc.py` extracts numeric public covariates, enforces `num_pub_features`, and drops the `cases` target column before writing `train_/test_` splits and metadata (`split_info`, `public_feature_map`).  
- `align_private_tensor` in `src/nyc_gradmeta/models/forecasting_gradmeta_nyc.py` right-aligns the loaded tensor, pads to match `[train + test]`, and applies `private_norm` (`zscore_time` by default).  
- `scripts/build_private_opentable_tensor.py` also prints the tensor shape/date range; the final `.pt` is `[16, T]` and loaded via `torch.load` (see training script lines referencing `private_dir`).

</details>

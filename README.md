# NYC GradMeta – Age-Structured COVID Forecasting (NYC, 2020–2022)

## Overview

This repository implements a clean, reproducible pipeline for:

- Age-structured SEIRM metapopulation modeling (16x16 contact matrix)
- NYC city-level COVID forecasting (2020–2022)
- Public signals: cases, deaths, Google Mobility, Google Trends
- Private signal: OpenTable reservation data
  The following are links for the datasets:
  https://trends.google.com/explore?geo=US-NY&q=Covid%252019&date=2020-01-01%202022-01-01
  https://www.google.com/covid19/mobility/
  https://www.kaggle.com/datasets/pizacd/opentable-reservation-data

The goal is to test whether adding OpenTable improves forecasting performance.

This repository implements a reproducible pipeline for NYC-level COVID forecasting using:

- NYC DOHMH daily counts (cases, deaths, hospitalizations, probable cases)
- Google Mobility (NYC counties aggregated to daily NYC mean)
- Google Trends (daily; downloaded programmatically via pytrends)
- OpenTable YoY seated diners (private signal; processed separately)

Goal: test whether adding OpenTable improves forecasting performance while keeping the core pipeline identical to the Bogota GradABM workflow.

---

## Data

### Folder structure

- `data/raw/` — downloaded raw inputs
- `data/processed/` — generated outputs (this repo tracks processed CSVs)

### Required raw files (place in `data/raw/`)

#### 1) NYC DOHMH daily COVID counts

Source: NYC Open Data — “COVID-19 Daily Counts of Cases, Hospitalizations, and Deaths” (CSV export)  
Save as (exact filename expected by scripts):

- `COVID-19_Daily_Counts_of_Cases,_Hospitalizations,_and_Deaths_20260213.csv`

(If you download a newer export, either rename it to match the expected filename or update the script.)

#### 2) Google Mobility zip

Source: Google COVID-19 Community Mobility Reports  
Save as:

- `Region_Mobility_Report_CSVs.zip`

This zip should contain (at least):

- `2020_US_Region_Mobility_Report.csv`
- `2021_US_Region_Mobility_Report.csv`
- `2022_US_Region_Mobility_Report.csv`

#### 3) Google Trends

No raw file required. Download happens via `pytrends` and writes directly to `data/processed/`.

#### 4) OpenTable YoY seated diners (processed separately)

Source: Kaggle OpenTable dataset  
Save as:

- `YoY_Seated_Diner_Data.csv`

### 5) Patchflow contact matrix and population

We model NYC using 16 age patches consistent with Prem et al. (2019) contact matrix groupings (0–4, 5–9, ..., 70–74, 75+). NYC age-stratified population counts are taken from the New York State Department of Health Vital Statistics 2020, Table 01 (New York City Population by Age): https://health.ny.gov/statistics/vital_statistics/2020/table01.htm
. The original age breakdowns were aggregated into the 16 SEIRM age bins (combining <1–4 into 0–4, 15–17 and 18–19 into 15–19, and 75–79, 80–84, and 85+ into 75+). The resulting processed file (data/processed/population_nyc_age16_2020.csv) contains the fixed population vector (total = 8,253,213) used as the age-stratified num_agents input to the SEIRM simulator.

### Data sources (links)

- Google Trends UI (reference only; do not export CSV manually): https://trends.google.com
- Google Mobility: https://www.google.com/covid19/mobility/
- OpenTable Kaggle dataset: https://www.kaggle.com/datasets/pizacd/opentable-reservation-data
- https://health.ny.gov/statistics/vital_statistics/2020/table01.htm
- https://github.com/NSSAC/patchflow-data/blob/main/data/v1.0_age_stratified/USA/USA_admin1_population_agestrat.patchsim

---

## Build the processed datasets

From repo root:

### One-command build (public data only)

```bash
./scripts/build_data.sh
```

This will populate `data/processed/` with:

- **NYC master daily** table (`nyc_master_daily.csv`) combining DOHMH counts, Google Mobility, and Google Trends.
- **OpenTable YoY daily** CSV, aligned to NYC dates (but not yet converted to the private tensor).

---

## Run the NYC forecasting pipeline (NN → SEIRM → error-correction adapter)

The full pipeline for a given ASOF date consists of:

1. **Neural parameter network (CalibNNTwoEncoderThreeOutputs)**  
   Learns weekly SEIRM epidemiological parameters, per-patch seed status, and a learned beta matrix from:
   - Public sequence: NYC cases + 1D mobility/trend feature (`pub_0`).
   - Private sequence: OpenTable per-patch signal (age-stratified via population shares).

2. **Mechanistic SEIRM simulator (MetapopulationSEIRMBeta)**  
   Runs a daily SEIRM metapopulation simulation with:
   - Age-stratified population vector (16 patches, consistent with the US contact matrix).
   - Contact / migration matrix (`contact_matrix_us.csv`).
   - Weekly parameters + beta matrix + seed status from the neural parameter network.

3. **Error-correction adapter (ErrorCorrectionAdapter)**  
   A lightweight GRU-based residual model that takes the **simulated city-wide daily cases** and learns an additive correction on top of the mechanistic forecast.  
   Training loss is computed on the **adapter-corrected** predictions:
   \[
   \text{preds} = \text{SEIRM\_preds} + \text{adapter(SEIRM\_preds)}
   \]

### One-command end-to-end run

From the repo root, after creating and activating your Python environment and installing dependencies:

```bash
chmod +x scripts/*.sh
./scripts/run_all.sh 2021-12-31
```

This runs:

- `scripts/build_data.sh` – builds public processed datasets.  
- `scripts/build_private_opentable_tensor.py` – converts OpenTable daily signal into a `[num_patch, T]` private tensor using the age-stratified population vector and writes `data/processed/private/opentable_private_lap_<ASOF>.pt`.  
- `scripts/prepare_online_nyc.py` – builds `data/processed/online/train_<ASOF>.csv` and `test_<ASOF>.csv`, with columns:
  - `cases` – NYC daily cases (target).  
  - `pub_0` – 1D public feature (mobility index or trend, controlled by `public_feature_mode` in `configs/nyc.json`).  
- `python -m nyc_gradmeta.models.forecasting_gradmeta_nyc` – trains the parameter NN + SEIRM + error-correction adapter on the training window and saves a 28-day forecast for the test window:
  - Numpy: `outputs/nyc/<ASOF>/forecast_28d.npy`  
  - CSV: `outputs/nyc/<ASOF>/forecast_28d.csv` (column `pred_cases`)

### Manual commands (if you prefer step-by-step)

Assuming you’re in the repo root, have installed this package in editable mode
(`pip install -e .` inside a virtualenv), and created all required raw data files:

```bash
# 1) Build public processed data
./scripts/build_data.sh

# 2) Build private OpenTable tensor (age-stratified, 16 patches)
python scripts/build_private_opentable_tensor.py \
  --config configs/nyc.json \
  --asof 2021-12-31

# 3) Build online train/test CSVs for the given ASOF
python scripts/prepare_online_nyc.py \
  --config configs/nyc.json \
  --asof 2021-12-31

# 4) Train and forecast with NN -> SEIRM -> error-correction adapter
python -m nyc_gradmeta.models.forecasting_gradmeta_nyc \
  --config configs/nyc.json \
  --asof 2021-12-31
```

You can change the ASOF date (`--asof`) and `week` multiplier (training horizon) in
`forecasting_gradmeta_nyc.py` via CLI:

```bash
python -m nyc_gradmeta.models.forecasting_gradmeta_nyc \
  --config configs/nyc.json \
  --asof 2021-12-31 \
  --week 8
```

This uses a training window of `train_days_base * week` days and always forecasts
the last `days_head` days (default 28).

---

## Three dataset modes (easy-to-use scripts)

The pipeline supports three configurations:

| Mode | Description | Script |
|------|-------------|--------|
| **1. Master only** | Public data only (cases, mobility, trends). No OpenTable. | `./scripts/run_nyc_master_only.sh <ASOF>` |
| **2. Master + OpenTable** | Public data + OpenTable private tensor (age-stratified). | `./scripts/run_nyc_with_opentable.sh <ASOF>` |
| **3. Privatized OpenTable** | (Planned) Same as 2 with differential privacy / Laplace noise on OpenTable. | _To be added._ |

All scripts expect the repo root as the current directory and use `configs/nyc.json`.  
Use the project’s virtualenv so `python` points to `.venv/bin/python`:

```bash
# One-time setup
python3 -m venv .venv
. .venv/bin/activate   # or: source .venv/bin/activate
pip install -e .
chmod +x scripts/*.sh
```

### Commands to run each pipeline

**1) Master only (baseline, no private data)**

```bash
. .venv/bin/activate
./scripts/run_nyc_master_only.sh 2022-10-15
```

This builds public data, prepares online train/test CSVs, then trains and forecasts with `--no_private`.  
Outputs: `outputs/nyc/2022-10-15/forecast_28d.npy`, `forecast_28d.csv`, `param_model.pt`, `error_adapter.pt`.

**2) Master + OpenTable**

```bash
. .venv/bin/activate
./scripts/run_nyc_with_opentable.sh 2022-10-15
```

This builds public data, builds the OpenTable private tensor from `data/processed/opentable_yoy_daily.csv`, prepares online train/test, then trains and forecasts with the private encoder.  
Outputs: same as above in `outputs/nyc/2022-10-15/`.

**3) Full pipeline (same as 2) via single script**

```bash
. .venv/bin/activate
./scripts/run_all.sh 2022-10-15
```

`run_all.sh` runs: build_data.sh → build_private_opentable_tensor.py (with `--opentable_csv` / `--opentable_col`) → prepare_online_nyc.py → forecasting_gradmeta_nyc.

---

## Visualization

After training, plot forecast vs. ground truth for the test window:

```bash
. .venv/bin/activate
python -m nyc_gradmeta.visualization --asof 2022-10-15 --config configs/nyc.json --mode master_opentable
```

Or use the wrapper script:

```bash
./scripts/run_visualization.sh 2022-10-15 master_opentable
./scripts/run_visualization.sh 2022-10-15 master_only
```

Figures are saved under `results/`, e.g. `results/nyc_forecast_master_opentable_2022-10-15.png`.  
If matplotlib reports a non-writable cache directory, set `MPLCONFIGDIR` to a writable path (e.g. `MPLCONFIGDIR=.venv/mplconfig`).

---

## Build private tensor (when OpenTable is not in master)

If the OpenTable series is not merged into `nyc_master_daily.csv`, pass the standalone CSV:

```bash
python scripts/build_private_opentable_tensor.py \
  --config configs/nyc.json \
  --asof 2022-10-15 \
  --opentable_csv data/processed/opentable_yoy_daily.csv \
  --opentable_col yoy_seated_diner
```

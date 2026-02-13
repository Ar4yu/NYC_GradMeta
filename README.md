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

### One-command build

```bash
./scripts/build_data.sh
```

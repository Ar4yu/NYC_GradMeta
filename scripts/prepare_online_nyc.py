import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json


def compute_mobility_index(df: pd.DataFrame) -> pd.Series:
    mob_cols = [c for c in df.columns if c.startswith("mob_")]
    if len(mob_cols) == 0:
        raise ValueError("No mobility columns found (expected columns starting with 'mob_').")
    # simple, stable 1D compression: mean across mobility signals
    return df[mob_cols].mean(axis=1)


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nyc.json")
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD. Test window ends on this date (inclusive).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    nyc = cfg["nyc"]

    master_path = Path(nyc["paths"]["master_daily_csv"])
    online_dir = Path(nyc["paths"]["online_dir"])
    online_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(master_path)
    if "date" not in df.columns:
        raise ValueError("master_daily_csv must include a 'date' column.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Build target column "cases" from configured source column
    src = nyc.get("target_source_column", "total_cases")
    if src not in df.columns:
        raise ValueError(f"Configured target_source_column '{src}' not found in master_daily_csv columns.")
    df["cases"] = df[src].astype(float)

    # Build a single public feature column to match num_pub_features=1
    mode = nyc.get("public_feature_mode", "mobility_index")
    if mode == "mobility_index":
        df["pub_0"] = compute_mobility_index(df).astype(float)
    elif mode == "trend":
        if "trend_covid_topic" not in df.columns:
            raise ValueError("Expected trend column 'trend_covid_topic' not found.")
        df["pub_0"] = df["trend_covid_topic"].astype(float)
    else:
        raise ValueError(f"Unknown public_feature_mode='{mode}'. Use 'mobility_index' or 'trend'.")

    # Keep only what the forecasting pipeline expects in the public CSV:
    # date column is okay to keep, but SeqDataset may read df.values; better to DROP it.
    # We'll drop 'date' in the artifact and rely on implicit ordering.
    keep_cols = ["cases"] + [f"pub_{i}" for i in range(nyc["num_pub_features"])]
    out = df[keep_cols].copy()

    # Split train/test by last 28 days ending at ASOF (inclusive)
    asof = pd.to_datetime(args.asof)
    test_days = int(nyc["test_days"])
    test_start = asof - pd.Timedelta(days=test_days - 1)

    # Restrict to <= asof
    mask_upto = df["date"] <= asof
    df_upto = df.loc[mask_upto].copy()
    out_upto = out.loc[mask_upto].reset_index(drop=True)

    # Find index split based on date range
    dates_upto = df_upto["date"].reset_index(drop=True)
    test_mask = (dates_upto >= test_start) & (dates_upto <= asof)
    if test_mask.sum() != test_days:
        raise ValueError(
            f"Expected exactly {test_days} test days between {test_start.date()} and {asof.date()}, "
            f"found {int(test_mask.sum())}. Check missing dates in master_daily_csv."
        )

    train_df = out_upto.loc[~test_mask].reset_index(drop=True)
    test_df = out_upto.loc[test_mask].reset_index(drop=True)

    # Save
    train_path = online_dir / f"train_{args.asof}.csv"
    test_path = online_dir / f"test_{args.asof}.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Wrote:")
    print(f"  {train_path}  (rows={len(train_df)})")
    print(f"  {test_path}   (rows={len(test_df)})")
    print("Columns:", list(train_df.columns))


if __name__ == "__main__":
    main()
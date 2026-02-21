import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_population_share(pop_path: Path, num_patch: int) -> np.ndarray:
    pop_df = pd.read_csv(pop_path)
    pop_col = None
    for col in ("population", "Population", "pop", "POPULATION"):
        if col in pop_df.columns:
            pop_col = col
            break
    if pop_col is None:
        raise ValueError(f"population CSV must have a population column. Columns={pop_df.columns.tolist()}")

    pop = pop_df[pop_col].to_numpy(dtype=np.float64)
    if len(pop) != num_patch:
        raise ValueError(f"Expected {num_patch} population rows, found {len(pop)}.")
    total = pop.sum()
    if total <= 0:
        raise ValueError("Population sum must be positive.")
    return pop / total


def load_master_dates(master_path: Path, asof: pd.Timestamp) -> pd.DatetimeIndex:
    master_df = pd.read_csv(master_path)
    if "date" not in master_df.columns:
        raise ValueError("master_daily_csv must include 'date'.")
    master_df["date"] = pd.to_datetime(master_df["date"])
    master_df = master_df.sort_values("date")
    master_df = master_df[master_df["date"] <= asof].reset_index(drop=True)
    if master_df.empty:
        raise ValueError(f"No master rows found up to asof={asof.date()}.")
    if master_df["date"].duplicated().any():
        raise ValueError("master_daily_csv has duplicate dates.")
    diffs = master_df["date"].diff().dropna().dt.days
    if not (diffs == 1).all():
        raise ValueError("master_daily_csv must be daily contiguous up to asof.")
    return pd.DatetimeIndex(master_df["date"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nyc.json")
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD. Build tensor up to this date (inclusive).")
    ap.add_argument(
        "--opentable_csv",
        default=None,
        help="Optional path to a processed OpenTable daily CSV. If omitted, we try to infer from master.",
    )
    ap.add_argument(
        "--opentable_col",
        default="opentable",
        help="Column name for the OpenTable signal (in opentable_csv or master_daily_csv).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    nyc = cfg["nyc"]

    master_path = Path(nyc["paths"]["master_daily_csv"])
    private_dir = Path(nyc["paths"]["private_dir"])
    private_dir.mkdir(parents=True, exist_ok=True)

    pop_path = Path(nyc["paths"]["population_csv"])
    P = int(nyc["num_patch"])
    pop_share = load_population_share(pop_path, P)

    asof = pd.to_datetime(args.asof)
    full_dates = load_master_dates(master_path, asof)

    # Load OpenTable series
    if args.opentable_csv is not None:
        ot = pd.read_csv(args.opentable_csv)
        if "date" not in ot.columns:
            raise ValueError("OpenTable CSV must have 'date' column.")
        ot["date"] = pd.to_datetime(ot["date"])
        if args.opentable_col not in ot.columns:
            raise ValueError(f"OpenTable CSV missing column '{args.opentable_col}'.")
        ot = ot.sort_values("date")
        ot = ot[ot["date"] <= asof].reset_index(drop=True)
        source = ot[["date", args.opentable_col]].copy()
    else:
        # Try to source from master file (if you merged it there already)
        df = pd.read_csv(master_path)
        if "date" not in df.columns:
            raise ValueError("master_daily_csv must include 'date'.")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df[df["date"] <= asof].reset_index(drop=True)
        if args.opentable_col not in df.columns:
            raise ValueError(
                f"master_daily_csv missing '{args.opentable_col}'. "
                f"Either add it to master, or pass --opentable_csv."
            )
        source = df[["date", args.opentable_col]].copy()

    source = source.drop_duplicates("date").set_index("date").sort_index()
    aligned = source.reindex(full_dates)
    observed_days = int(aligned[args.opentable_col].notna().sum())
    missing_days = len(aligned) - observed_days

    # Fill missing dates so tensor length aligns with public series length.
    # Interpolate over time, then forward/back fill residual gaps, then zero any remaining.
    aligned[args.opentable_col] = (
        aligned[args.opentable_col]
        .astype(float)
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    series = aligned[args.opentable_col].to_numpy(dtype=np.float32)
    if np.isnan(series).any():
        raise ValueError("OpenTable aligned series contains NaNs after filling.")
    T = len(series)

    # Build [P, T] tensor (float32). Replicate with population-weight scaling.
    base = series[None, :]  # [1, T]
    weights = pop_share.astype(np.float32)[:, None]  # [P, 1]
    private = weights * base  # [P, T]

    # Optional: normalize to roughly [0,1] range for stability (keep contract simple)
    # If your professor pipeline already normalizes inside encoders, you can remove this.
    denom = np.max(np.abs(private)) if np.max(np.abs(private)) > 0 else 1.0
    private = private / denom

    tensor = torch.tensor(private, dtype=torch.float32)  # [P, T]

    out_path = private_dir / f"opentable_private_lap_{args.asof}.pt"
    torch.save(tensor, out_path)

    print("Wrote:", out_path)
    print("Tensor shape:", tuple(tensor.shape), "[num_patch, T]")
    print("Date range:", full_dates[0].date(), "→", full_dates[-1].date())
    print("Observed OpenTable days before fill:", observed_days)


if __name__ == "__main__":
    main()

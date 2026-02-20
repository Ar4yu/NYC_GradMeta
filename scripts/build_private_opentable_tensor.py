import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return json.load(f)


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
    pop_df = pd.read_csv(pop_path)
    if "population" not in pop_df.columns:
        raise ValueError("population CSV must have column 'population'.")
    pop = pop_df["population"].to_numpy(dtype=np.float64)
    if len(pop) != int(nyc["num_patch"]):
        raise ValueError(f"Expected {nyc['num_patch']} population rows, found {len(pop)}.")
    pop_share = pop / pop.sum()

    asof = pd.to_datetime(args.asof)

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
        series = ot[args.opentable_col].astype(float).to_numpy()
        dates = ot["date"]
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
        series = df[args.opentable_col].astype(float).to_numpy()
        dates = df["date"]

    # Sanity: no NaNs
    if np.isnan(series).any():
        raise ValueError("OpenTable series contains NaNs. Interpolate/fill in preprocessing first.")

    T = len(series)
    P = int(nyc["num_patch"])

    # Build [P, T] tensor (float32). Replicate with population-weight scaling.
    base = series.astype(np.float32)[None, :]            # [1, T]
    weights = pop_share.astype(np.float32)[:, None]      # [P, 1]
    private = weights * base                             # [P, T]

    # Optional: normalize to roughly [0,1] range for stability (keep contract simple)
    # If your professor pipeline already normalizes inside encoders, you can remove this.
    denom = np.max(np.abs(private)) if np.max(np.abs(private)) > 0 else 1.0
    private = private / denom

    tensor = torch.tensor(private, dtype=torch.float32)  # [P, T]

    out_path = private_dir / f"opentable_private_lap_{args.asof}.pt"
    torch.save(tensor, out_path)

    print("Wrote:", out_path)
    print("Tensor shape:", tuple(tensor.shape), "[num_patch, T]")
    print("Date range:", dates.iloc[0].date(), "→", dates.iloc[-1].date())


if __name__ == "__main__":
    main()
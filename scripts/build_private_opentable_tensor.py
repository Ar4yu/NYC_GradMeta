import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json

from nyc_gradmeta.utils import private_artifact_stem


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


def load_observed_source(
    csv_path: Path,
    value_col: str,
    asof: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError(f"{csv_path} must include 'date'.")
    if value_col not in df.columns:
        raise ValueError(f"{csv_path} missing '{value_col}'. Columns={df.columns.tolist()}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    observed = df[df[value_col].notna()].copy()
    if observed.empty:
        raise ValueError(f"{csv_path} has no observed rows for '{value_col}'.")
    observed_start = observed["date"].iloc[0]
    observed_end = observed["date"].iloc[-1]
    source = observed[observed["date"] <= asof][["date", value_col]].copy()
    return source, observed_start, observed_end


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
    ap.add_argument(
        "--matched_window_with_opentable",
        action="store_true",
        help=(
            "Clip the private tensor to the true observed public/OpenTable overlap window. "
            "This is the fair A/B experiment mode."
        ),
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
    full_dates_all = load_master_dates(master_path, asof)
    public_start = full_dates_all.min()
    public_end = full_dates_all.max()

    # Load OpenTable series
    if args.opentable_csv is not None:
        source, observed_start, observed_end = load_observed_source(
            Path(args.opentable_csv),
            args.opentable_col,
            asof,
        )
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
        observed = df[df[args.opentable_col].notna()].copy()
        if observed.empty:
            raise ValueError(f"master_daily_csv has no observed values for '{args.opentable_col}'.")
        observed_start = observed["date"].iloc[0]
        observed_end = observed["date"].iloc[-1]
        source = observed[["date", args.opentable_col]].copy()

    requested_asof = asof
    if args.matched_window_with_opentable:
        joint_start = max(public_start, observed_start)
        joint_end = min(public_end, observed_end, requested_asof)
        if joint_end < joint_start:
            raise ValueError(
                f"No joint public/OpenTable overlap for requested asof={args.asof}. "
                f"public=[{public_start.date()}, {public_end.date()}], "
                f"opentable=[{observed_start.date()}, {observed_end.date()}]."
            )
        full_dates = full_dates_all[(full_dates_all >= joint_start) & (full_dates_all <= joint_end)]
        actual_asof = joint_end
    else:
        if requested_asof > observed_end:
            raise ValueError(
                f"Requested ASOF {requested_asof.date()} exceeds the last observed OpenTable date "
                f"{observed_end.date()}. Refusing to build synthetic future OpenTable data. "
                "Use --matched_window_with_opentable to clip to the true observed overlap window."
            )
        full_dates = full_dates_all
        actual_asof = requested_asof
        joint_start = public_start
        joint_end = requested_asof

    source = source.drop_duplicates("date").set_index("date").sort_index()
    aligned = source.reindex(full_dates)
    observed_days = int(aligned[args.opentable_col].notna().sum())
    missing_days = len(aligned) - observed_days

    # Fill only inside the observed window used for this experiment.
    # This allows interpolation for interior gaps, but never carries OpenTable into future dates.
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

    out_base = private_artifact_stem(
        args.asof,
        matched_window_with_opentable=args.matched_window_with_opentable,
    )
    out_path = private_dir / f"{out_base}.pt"
    torch.save(tensor, out_path)
    meta = {
        "requested_asof": args.asof,
        "actual_asof": str(actual_asof.date()),
        "matched_window_with_opentable": bool(args.matched_window_with_opentable),
        "laplace_noise_applied": False,
        "public_start": str(public_start.date()),
        "public_end": str(public_end.date()),
        "opentable_observed_start": str(observed_start.date()),
        "opentable_observed_end": str(observed_end.date()),
        "joint_start": str(joint_start.date()),
        "joint_end": str(joint_end.date()),
        "observed_days_in_tensor_window": observed_days,
        "filled_days_in_tensor_window": missing_days,
        "tensor_shape": list(tensor.shape),
    }
    meta_path = private_dir / f"{out_base}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:", out_path)
    print("Metadata:", meta_path)
    print("Tensor shape:", tuple(tensor.shape), "[num_patch, T]")
    print("Date range:", full_dates[0].date(), "→", full_dates[-1].date())
    print("Observed OpenTable days before fill:", observed_days)
    print("Filled days inside experiment window:", missing_days)
    if args.matched_window_with_opentable:
        print("[info] Matched-window mode enabled. No OpenTable dates beyond the observed range were used.")
    else:
        print("[info] Non-matched mode used only because requested ASOF stayed within observed OpenTable dates.")


if __name__ == "__main__":
    main()

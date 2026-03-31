import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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


def gaussian_multiplier(delta: float) -> float:
    if not (0.0 < float(delta) < 1.0):
        raise ValueError(f"delta must be in (0,1), got {delta}.")
    return float(np.sqrt(2.0 * np.log(1.25 / float(delta))))


def resolve_dp_metadata(
    privacy_mode: str,
    mechanism: str,
    epsilon: float | None,
    delta: float,
    tmax: float,
    denominator_d: float,
    k_days: int,
    clipping_bound_pp: float,
    dp_seed: int,
) -> dict:
    mode = str(privacy_mode).strip().lower()
    mech = str(mechanism).strip().lower()
    if mode in {"", "none", "nonprivate", "non_private"}:
        return {
            "privacy_mode": "none",
            "mechanism": "none",
            "epsilon": None,
            "delta": None,
            "Tmax": None,
            "D": None,
            "K": int(k_days),
            "clipping_bound_pp": float(clipping_bound_pp),
            "sensitivity_day_pp": None,
            "sensitivity_l2_pp": None,
            "sigma_pp": None,
            "dp_seed": None,
            "noise_added": False,
        }

    if mech != "gaussian":
        raise ValueError(f"Unsupported mechanism '{mechanism}'. Expected gaussian.")
    if mode not in {"event", "restaurant"}:
        raise ValueError(f"Unsupported privacy_mode '{privacy_mode}'. Expected none|event|restaurant.")
    if epsilon is None or float(epsilon) <= 0:
        raise ValueError(f"epsilon must be positive for DP runs, got {epsilon}.")
    if float(tmax) <= 0 or float(denominator_d) <= 0:
        raise ValueError(f"Tmax and D must be positive, got Tmax={tmax}, D={denominator_d}.")
    if int(k_days) <= 0:
        raise ValueError(f"K must be positive, got {k_days}.")
    if float(clipping_bound_pp) <= 0:
        raise ValueError(f"clipping_bound_pp must be positive, got {clipping_bound_pp}.")

    delta_day = (float(tmax) / float(denominator_d)) * 100.0
    if mode == "event":
        delta_l2 = delta_day
    else:
        delta_l2 = delta_day * float(np.sqrt(float(k_days)))
    sigma_pp = (delta_l2 / float(epsilon)) * gaussian_multiplier(delta)
    return {
        "privacy_mode": mode,
        "mechanism": mech,
        "epsilon": float(epsilon),
        "delta": float(delta),
        "Tmax": float(tmax),
        "D": float(denominator_d),
        "K": int(k_days),
        "clipping_bound_pp": float(clipping_bound_pp),
        "sensitivity_day_pp": float(delta_day),
        "sensitivity_l2_pp": float(delta_l2),
        "sigma_pp": float(sigma_pp),
        "dp_seed": int(dp_seed),
        "noise_added": True,
    }


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
    ap.add_argument(
        "--privacy_mode",
        choices=["none", "event", "restaurant"],
        default="none",
        help="Privacy mode for the city-level OpenTable series before patch lifting.",
    )
    ap.add_argument(
        "--mechanism",
        choices=["gaussian"],
        default="gaussian",
        help="Differential privacy mechanism used when privacy_mode is not none.",
    )
    ap.add_argument("--epsilon", type=float, default=None, help="DP epsilon for Gaussian runs.")
    ap.add_argument("--delta", type=float, default=1e-4, help="DP delta for Gaussian runs.")
    ap.add_argument("--tmax", type=float, default=200.0, help="Maximum daily contribution clip at the event level.")
    ap.add_argument("--denominator_d", type=float, default=80000.0, help="Daily denominator D in percentage-point sensitivity.")
    ap.add_argument(
        "--clipping_bound_pp",
        type=float,
        default=100.0,
        help="Fixed symmetric clipping bound in percentage points for the OpenTable YoY series.",
    )
    ap.add_argument(
        "--dp_seed",
        type=int,
        default=0,
        help="Random seed used for Gaussian DP noise generation.",
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

    series_observed_pp = aligned[args.opentable_col].to_numpy(dtype=np.float32)
    if np.isnan(series_observed_pp).any():
        raise ValueError("OpenTable aligned series contains NaNs after filling.")
    series_clipped_pp = np.clip(
        series_observed_pp,
        -float(args.clipping_bound_pp),
        float(args.clipping_bound_pp),
    ).astype(np.float32)
    T = len(series_clipped_pp)
    dp_meta = resolve_dp_metadata(
        privacy_mode=args.privacy_mode,
        mechanism=args.mechanism,
        epsilon=args.epsilon,
        delta=args.delta,
        tmax=args.tmax,
        denominator_d=args.denominator_d,
        k_days=T,
        clipping_bound_pp=args.clipping_bound_pp,
        dp_seed=args.dp_seed,
    )

    if dp_meta["privacy_mode"] == "none":
        noise_pp = np.zeros(T, dtype=np.float32)
        series_released_pp = series_clipped_pp.copy()
    else:
        rng = np.random.default_rng(int(args.dp_seed))
        noise_pp = rng.normal(loc=0.0, scale=float(dp_meta["sigma_pp"]), size=T).astype(np.float32)
        series_noised_pp = series_clipped_pp + noise_pp
        # Post-processing clip keeps the downstream representation bounded with a fixed public rule.
        series_released_pp = np.clip(
            series_noised_pp,
            -float(args.clipping_bound_pp),
            float(args.clipping_bound_pp),
        ).astype(np.float32)

    # Fixed scaling keeps the private input range documented and independent of raw data extrema.
    series_scaled = (series_released_pp / float(args.clipping_bound_pp)).astype(np.float32)

    # Build [P, T] tensor (float32). Replicate with population-weight scaling.
    base = series_scaled[None, :]  # [1, T]
    weights = pop_share.astype(np.float32)[:, None]  # [P, 1]
    private = weights * base  # [P, T]

    tensor = torch.tensor(private, dtype=torch.float32)  # [P, T]

    out_base = private_artifact_stem(
        args.asof,
        matched_window_with_opentable=args.matched_window_with_opentable,
        privacy_mode=args.privacy_mode,
        mechanism=args.mechanism,
        epsilon=args.epsilon,
    )
    out_path = private_dir / f"{out_base}.pt"
    torch.save(tensor, out_path)
    meta = {
        "requested_asof": args.asof,
        "actual_asof": str(actual_asof.date()),
        "matched_window_with_opentable": bool(args.matched_window_with_opentable),
        "laplace_noise_applied": False,
        "noise_added_before_patch_lifting": True,
        "scaling_rule": "fixed_divide_by_clipping_bound_pp",
        "post_noise_clip_applied": True,
        "public_start": str(public_start.date()),
        "public_end": str(public_end.date()),
        "opentable_observed_start": str(observed_start.date()),
        "opentable_observed_end": str(observed_end.date()),
        "joint_start": str(joint_start.date()),
        "joint_end": str(joint_end.date()),
        "observed_days_in_tensor_window": observed_days,
        "filled_days_in_tensor_window": missing_days,
        "series_csv": str((private_dir / f"{out_base}_series.csv").as_posix()),
        "tensor_shape": list(tensor.shape),
    }
    meta.update(dp_meta)
    meta_path = private_dir / f"{out_base}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    signal_df = pd.DataFrame(
        {
            "date": pd.Index(full_dates).strftime("%Y-%m-%d"),
            "observed_or_filled_yoy_pp": series_observed_pp,
            "clipped_yoy_pp": series_clipped_pp,
            "noise_pp": noise_pp,
            "released_yoy_pp": series_released_pp,
            "scaled_signal": series_scaled,
        }
    )
    signal_path = private_dir / f"{out_base}_series.csv"
    signal_df.to_csv(signal_path, index=False)

    print("Wrote:", out_path)
    print("Metadata:", meta_path)
    print("Signal CSV:", signal_path)
    print("Tensor shape:", tuple(tensor.shape), "[num_patch, T]")
    print("Date range:", full_dates[0].date(), "→", full_dates[-1].date())
    print("Observed OpenTable days before fill:", observed_days)
    print("Filled days inside experiment window:", missing_days)
    if dp_meta["privacy_mode"] != "none":
        print(
            "[info] Applied Gaussian DP:",
            f"mode={dp_meta['privacy_mode']}, eps={dp_meta['epsilon']}, delta={dp_meta['delta']},",
            f"K={dp_meta['K']}, sigma_pp={dp_meta['sigma_pp']:.6f}",
        )
    else:
        print(
            "[info] Non-private OpenTable build with fixed clipping/scaling:",
            f"clipping_bound_pp={args.clipping_bound_pp}",
        )
    if args.matched_window_with_opentable:
        print("[info] Matched-window mode enabled. No OpenTable dates beyond the observed range were used.")
    else:
        print("[info] Non-matched mode used only because requested ASOF stayed within observed OpenTable dates.")


if __name__ == "__main__":
    main()

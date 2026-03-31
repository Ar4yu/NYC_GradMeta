import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".venv/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nyc_gradmeta.utils import online_artifact_stem, run_tag_for_mode, smoothing_label


def load_forecast(
    asof: str,
    smooth_cases_window: int,
    run_tag: str | None = None,
    root: Path = Path("."),
) -> pd.DataFrame:
    out_dir = root / "outputs" / "nyc" / asof
    forecast_base = f"forecast_28d_w{int(smooth_cases_window)}"
    candidates = []
    if run_tag:
        candidates.append((out_dir / f"{forecast_base}_{run_tag}.csv", out_dir / f"{forecast_base}_{run_tag}.npy"))
    candidates.append((out_dir / f"{forecast_base}.csv", out_dir / f"{forecast_base}.npy"))

    for csv_path, npy_path in candidates:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "pred_cases" not in df.columns:
                raise ValueError(f"{csv_path} must have column 'pred_cases'.")
            return df
        if npy_path.exists():
            preds = np.load(npy_path)
            return pd.DataFrame(
                {
                    "day_idx": np.arange(len(preds), dtype=int),
                    "pred_cases": preds,
                    "smooth_cases_window": np.full(len(preds), int(smooth_cases_window), dtype=int),
                }
            )
    raise FileNotFoundError(f"Could not find forecast file in {out_dir}")


def plot_forecast_vs_truth(
    asof: str,
    test_csv: Path,
    run_tag: str,
    out_path: Path,
    smooth_cases_window: int,
    title: str = "",
    subtitle: str = "",
) -> None:
    test_df = pd.read_csv(test_csv)
    if "cases" not in test_df.columns:
        raise ValueError(f"{test_csv} must have column 'cases'.")

    y_true = test_df["cases"].to_numpy(dtype=float)
    forecast_df = load_forecast(asof, smooth_cases_window=smooth_cases_window, run_tag=run_tag)
    y_pred = forecast_df["pred_cases"].to_numpy(dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: truth len={len(y_true)}, forecast len={len(y_pred)}. "
            "Ensure test_days == forecast horizon."
        )

    x = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_true, label=f"Truth ({smoothing_label(smooth_cases_window)})", marker="o")
    ax.plot(x, y_pred, label=f"Forecast ({smoothing_label(smooth_cases_window)})", marker="x")
    ax.set_xlabel("Day in forecast window")
    ax.set_ylabel("Daily cases")
    ax.legend(loc="upper right")
    fig.suptitle(
        title or f"NYC Forecast vs Truth: {run_tag}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=10, color="dimgray")
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def load_metrics_json(asof: str, run_tag: str, root: Path = Path(".")) -> dict | None:
    path = root / "outputs" / "nyc" / asof / f"metrics_{run_tag}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="ASOF date used for forecast (YYYY-MM-DD).")
    ap.add_argument(
        "--config",
        default="configs/nyc.json",
        help="Config JSON (used to locate online_dir/test CSV).",
    )
    ap.add_argument("--smooth_cases_window", type=int, choices=[0, 3, 7], default=0)
    ap.add_argument("--window_days", type=int, default=170)
    ap.add_argument(
        "--mode",
        choices=["master_only", "master_opentable"],
        default="master_opentable",
        help="Label used only for output filename.",
    )
    ap.add_argument("--use_adapter", action="store_true", help="Load the adapter run tag outputs.")
    ap.add_argument(
        "--matched_window_with_opentable",
        action="store_true",
        help="Use matched-window split metadata and run-tag-specific forecast artifacts.",
    )
    ap.add_argument(
        "--privacy_mode",
        choices=["none", "event", "restaurant"],
        default="none",
        help="Privacy mode for DP OpenTable run tags.",
    )
    ap.add_argument(
        "--mechanism",
        choices=["gaussian"],
        default="gaussian",
        help="Privacy mechanism for DP OpenTable run tags.",
    )
    ap.add_argument("--epsilon", type=float, default=None, help="Privacy epsilon for DP OpenTable run tags.")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)
    nyc = cfg["nyc"]
    online_dir = Path(nyc["paths"]["online_dir"])
    mode = "public_only" if args.mode == "master_only" else "public_opentable"
    run_tag = run_tag_for_mode(
        mode=mode,
        use_adapter=bool(args.use_adapter),
        smooth_cases_window=args.smooth_cases_window,
        matched_window_with_opentable=args.matched_window_with_opentable,
        privacy_mode=args.privacy_mode,
        mechanism=args.mechanism,
        epsilon=args.epsilon,
    )
    test_csv = online_dir / (
        f"{online_artifact_stem('test', args.asof, args.window_days, args.smooth_cases_window, matched_window_with_opentable=args.matched_window_with_opentable)}.csv"
    )
    split_info_path = online_dir / (
        f"{online_artifact_stem('split_info', args.asof, args.window_days, args.smooth_cases_window, matched_window_with_opentable=args.matched_window_with_opentable)}.json"
    )
    with open(split_info_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    metrics = load_metrics_json(args.asof, run_tag)
    privacy_bits = []
    if metrics is not None and metrics.get("privacy_mode") not in (None, "", "none"):
        privacy_bits.append(f"{metrics.get('mechanism')} {metrics.get('privacy_mode')} DP")
        privacy_bits.append(f"eps={metrics.get('epsilon')}")
        if metrics.get("sigma_pp") is not None:
            privacy_bits.append(f"sigma={float(metrics['sigma_pp']):.3f} pp")

    out_path = Path("outputs") / "nyc" / args.asof / f"forecast_vs_truth_{run_tag}.png"
    plot_forecast_vs_truth(
        asof=args.asof,
        test_csv=test_csv,
        run_tag=run_tag,
        out_path=out_path,
        smooth_cases_window=args.smooth_cases_window,
        title=f"NYC Forecast vs Truth: {run_tag}",
        subtitle=(
            f"Window {split_info.get('window_start')} to {split_info.get('window_end')} | "
            f"Test {split_info.get('test_start')} to {split_info.get('test_end')} | "
            f"Requested ASOF {args.asof} | {smoothing_label(args.smooth_cases_window)}"
            + (f" | {' | '.join(privacy_bits)}" if privacy_bits else "")
        ),
    )
    print("Saved visualization to:", out_path)


if __name__ == "__main__":
    main()

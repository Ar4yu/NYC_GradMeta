import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".venv/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nyc_gradmeta.utils import online_artifact_stem, smoothing_label


def load_forecast(asof: str, smooth_cases_window: int, root: Path = Path(".")) -> pd.DataFrame:
    out_dir = root / "outputs" / "nyc" / asof
    csv_path = out_dir / f"forecast_28d_w{int(smooth_cases_window)}.csv"
    npy_path = out_dir / f"forecast_28d_w{int(smooth_cases_window)}.npy"

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
    out_path: Path,
    smooth_cases_window: int,
    title: str = "",
) -> None:
    test_df = pd.read_csv(test_csv)
    if "cases" not in test_df.columns:
        raise ValueError(f"{test_csv} must have column 'cases'.")

    y_true = test_df["cases"].to_numpy(dtype=float)
    forecast_df = load_forecast(asof, smooth_cases_window=smooth_cases_window)
    y_pred = forecast_df["pred_cases"].to_numpy(dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: truth len={len(y_true)}, forecast len={len(y_pred)}. "
            "Ensure test_days == forecast horizon."
        )

    x = np.arange(len(y_true))

    plt.figure(figsize=(8, 4))
    plt.plot(x, y_true, label=f"Truth ({smoothing_label(smooth_cases_window)})", marker="o")
    plt.plot(x, y_pred, label=f"Forecast ({smoothing_label(smooth_cases_window)})", marker="x")
    plt.xlabel("Day in forecast window")
    plt.ylabel("Daily cases")
    plt.title(title or f"NYC forecast vs truth (ASOF={asof}, {smoothing_label(smooth_cases_window)})")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


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
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)
    nyc = cfg["nyc"]
    online_dir = Path(nyc["paths"]["online_dir"])
    test_csv = online_dir / f"{online_artifact_stem('test', args.asof, args.window_days, args.smooth_cases_window)}.csv"

    out_path = Path("outputs") / "nyc" / args.asof / f"forecast_vs_truth_{args.mode}_w{args.smooth_cases_window}.png"
    plot_forecast_vs_truth(
        asof=args.asof,
        test_csv=test_csv,
        out_path=out_path,
        smooth_cases_window=args.smooth_cases_window,
        title=f"NYC forecast vs truth ({args.mode}, ASOF={args.asof}, {smoothing_label(args.smooth_cases_window)})",
    )
    print("Saved visualization to:", out_path)


if __name__ == "__main__":
    main()

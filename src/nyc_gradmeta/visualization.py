import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".venv/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_forecast(asof: str, root: Path = Path(".")) -> np.ndarray:
    out_dir = root / "outputs" / "nyc" / asof
    npy_path = out_dir / "forecast_28d.npy"
    csv_path = out_dir / "forecast_28d.csv"

    if npy_path.exists():
        return np.load(npy_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "pred_cases" not in df.columns:
            raise ValueError(f"{csv_path} must have column 'pred_cases'.")
        return df["pred_cases"].to_numpy(dtype=float)
    raise FileNotFoundError(f"Could not find forecast file in {out_dir}")




def plot_forecast_vs_truth(
    asof: str,
    test_csv: Path,
    out_path: Path,
    title: str = "",
) -> None:
    test_df = pd.read_csv(test_csv)
    if "cases" not in test_df.columns:
        raise ValueError(f"{test_csv} must have column 'cases'.")

    y_true = test_df["cases"].to_numpy(dtype=float)
    y_pred = load_forecast(asof)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: truth len={len(y_true)}, forecast len={len(y_pred)}. "
            "Ensure test_days == forecast horizon."
        )

    x = np.arange(len(y_true))

    plt.figure(figsize=(8, 4))
    plt.plot(x, y_true, label="True cases", marker="o")
    plt.plot(x, y_pred, label="Forecast cases", marker="x")
    plt.xlabel("Day in forecast window")
    plt.ylabel("Daily cases")
    plt.title(title or f"NYC forecast vs truth (ASOF={asof})")
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
    test_csv = online_dir / f"test_{args.asof}.csv"

    out_path = Path("results") / f"nyc_forecast_{args.mode}_{args.asof}.png"
    plot_forecast_vs_truth(
        asof=args.asof,
        test_csv=test_csv,
        out_path=out_path,
        title=f"NYC forecast vs truth ({args.mode}, ASOF={args.asof})",
    )
    print("Saved visualization to:", out_path)


if __name__ == "__main__":
    main()

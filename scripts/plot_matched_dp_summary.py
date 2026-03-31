import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".venv/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from nyc_gradmeta.utils import private_artifact_stem, run_tag_for_mode


EPSILONS = [1, 2, 4, 8, 16]


def load_metrics(out_dir: Path, run_tag: str) -> dict | None:
    path = out_dir / f"metrics_{run_tag}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_summary_rows(asof: str, out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    run_specs = [
        {
            "run_tag": run_tag_for_mode(
                mode="public_only",
                use_adapter=True,
                smooth_cases_window=7,
                matched_window_with_opentable=True,
            ),
            "label": "public_only",
        },
        {
            "run_tag": run_tag_for_mode(
                mode="public_opentable",
                use_adapter=True,
                smooth_cases_window=7,
                matched_window_with_opentable=True,
            ),
            "label": "public_opentable_nonprivate",
        },
    ]
    for privacy_mode in ("event", "restaurant"):
        for epsilon in EPSILONS:
            run_specs.append(
                {
                    "run_tag": run_tag_for_mode(
                        mode="public_opentable",
                        use_adapter=True,
                        smooth_cases_window=7,
                        matched_window_with_opentable=True,
                        privacy_mode=privacy_mode,
                        mechanism="gaussian",
                        epsilon=epsilon,
                    ),
                    "label": f"public_opentable_dp_gaussian_{privacy_mode}_eps{epsilon}",
                }
            )

    for spec in run_specs:
        metrics = load_metrics(out_dir, spec["run_tag"])
        if metrics is None:
            continue
        row = {
            "run_name": spec["run_tag"],
            "privacy_mode": metrics.get("privacy_mode", "none"),
            "mechanism": metrics.get("mechanism", "none"),
            "epsilon": metrics.get("epsilon"),
            "delta": metrics.get("delta"),
            "Tmax": metrics.get("Tmax"),
            "D": metrics.get("D"),
            "K": metrics.get("K"),
            "clipping_bound_pp": metrics.get("clipping_bound_pp"),
            "sensitivity_day_pp": metrics.get("sensitivity_day_pp"),
            "sensitivity_l2_pp": metrics.get("sensitivity_l2_pp"),
            "sigma_pp": metrics.get("sigma_pp"),
            "smoothing_window": metrics.get("smooth_cases_window"),
            "window_start": metrics.get("window_start"),
            "window_end": metrics.get("window_end"),
            "train_start": metrics.get("train_start"),
            "train_end": metrics.get("train_end"),
            "test_start": metrics.get("test_start"),
            "test_end": metrics.get("test_end"),
            "MSE": metrics.get("test_metrics", {}).get("mse"),
            "RMSE": metrics.get("test_metrics", {}).get("rmse"),
            "MAE": metrics.get("test_metrics", {}).get("mae"),
            "MAPE": metrics.get("test_metrics", {}).get("mape"),
        }
        rows.append(row)
    return rows


def plot_rmse_curve(df: pd.DataFrame, privacy_mode: str, out_path: Path) -> None:
    sub = df[df["privacy_mode"] == privacy_mode].sort_values("epsilon")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["epsilon"], sub["RMSE"], marker="o", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Test RMSE")
    ax.set_title(f"NYC Matched OpenTable Gaussian DP ({privacy_mode})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rmse_comparison(df: pd.DataFrame, out_path: Path) -> None:
    event_df = df[df["privacy_mode"] == "event"].sort_values("epsilon")
    rest_df = df[df["privacy_mode"] == "restaurant"].sort_values("epsilon")
    nonprivate = df[df["run_name"] == "public_opentable_adapter_w7_matched_ot"]
    public_only = df[df["run_name"] == "public_only_adapter_w7_matched_ot"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if not event_df.empty:
        ax.plot(event_df["epsilon"], event_df["RMSE"], marker="o", linewidth=2, label="Event-level Gaussian DP")
    if not rest_df.empty:
        ax.plot(rest_df["epsilon"], rest_df["RMSE"], marker="s", linewidth=2, label="Restaurant-level Gaussian DP")
    if not nonprivate.empty:
        ax.axhline(float(nonprivate["RMSE"].iloc[0]), linestyle="--", color="tab:green", label="Non-private OpenTable")
    if not public_only.empty:
        ax.axhline(float(public_only["RMSE"].iloc[0]), linestyle=":", color="black", label="Public-only baseline")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Test RMSE")
    ax.set_title("NYC Matched OpenTable Utility Comparison")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_signal_comparison(asof: str, private_dir: Path, out_path: Path) -> None:
    baseline_series = private_dir / f"{private_artifact_stem(asof, matched_window_with_opentable=True)}_series.csv"
    event_series = private_dir / (
        f"{private_artifact_stem(asof, matched_window_with_opentable=True, privacy_mode='event', mechanism='gaussian', epsilon=1)}_series.csv"
    )
    restaurant_series = private_dir / (
        f"{private_artifact_stem(asof, matched_window_with_opentable=True, privacy_mode='restaurant', mechanism='gaussian', epsilon=1)}_series.csv"
    )
    if not (baseline_series.exists() and event_series.exists() and restaurant_series.exists()):
        return

    base_df = pd.read_csv(baseline_series)
    event_df = pd.read_csv(event_series)
    rest_df = pd.read_csv(restaurant_series)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(base_df["date"], base_df["released_yoy_pp"], linewidth=2, label="Original / non-private")
    ax.plot(event_df["date"], event_df["released_yoy_pp"], linewidth=1.8, label="Event DP eps=1")
    ax.plot(rest_df["date"], rest_df["released_yoy_pp"], linewidth=1.8, label="Restaurant DP eps=1")
    tick_idx = max(1, len(base_df) // 8)
    ax.set_xticks(base_df["date"].iloc[::tick_idx])
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("OpenTable YoY seated-diner signal (pp)")
    ax.set_title("NYC Matched OpenTable Signal: Original vs Gaussian DP")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="Resolved matched-window ASOF used in outputs.")
    ap.add_argument("--config", default="configs/nyc.json")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    private_dir = Path(cfg["nyc"]["paths"]["private_dir"])
    out_dir = Path("outputs") / "nyc" / args.asof
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(build_summary_rows(args.asof, out_dir))
    df.to_csv(out_dir / "metrics_summary_dp_w7.csv", index=False)

    plot_rmse_curve(df, "event", out_dir / "rmse_vs_epsilon_event_w7_matched_ot.png")
    plot_rmse_curve(df, "restaurant", out_dir / "rmse_vs_epsilon_restaurant_w7_matched_ot.png")
    plot_rmse_comparison(df, out_dir / "rmse_vs_epsilon_comparison_w7_matched_ot.png")
    plot_signal_comparison(args.asof, private_dir, out_dir / "opentable_signal_dp_comparison_eps1_w7_matched_ot.png")

    print("Saved DP metrics summary:", out_dir / "metrics_summary_dp_w7.csv")


if __name__ == "__main__":
    main()

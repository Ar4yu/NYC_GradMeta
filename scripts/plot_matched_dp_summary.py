import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".venv/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from nyc_gradmeta.utils import private_artifact_stem, run_tag_for_mode


EPSILONS = [1, 2, 4, 8, 16]
BASELINE_RUNS = [
    "public_only_adapter_w7_matched_ot",
    "public_opentable_adapter_w7_matched_ot",
]
BASELINE_LABELS = {
    "public_only_adapter_w7_matched_ot": "Public-only",
    "public_opentable_adapter_w7_matched_ot": "OpenTable",
}
BASELINE_COLORS = {
    "public_only_adapter_w7_matched_ot": "#555555",
    "public_opentable_adapter_w7_matched_ot": "#1f77b4",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "experiment"


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
            "seed": metrics.get("seed"),
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


def write_single_seed_summary(asof: str, out_dir: Path, private_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(build_summary_rows(asof, out_dir))
    df.to_csv(out_dir / "metrics_summary_dp_w7.csv", index=False)

    plot_rmse_curve(df, "event", out_dir / "rmse_vs_epsilon_event_w7_matched_ot.png")
    plot_rmse_curve(df, "restaurant", out_dir / "rmse_vs_epsilon_restaurant_w7_matched_ot.png")
    plot_rmse_comparison(df, out_dir / "rmse_vs_epsilon_comparison_w7_matched_ot.png")
    plot_signal_comparison(asof, private_dir, out_dir / "opentable_signal_dp_comparison_eps1_w7_matched_ot.png")
    return df


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


def plot_aggregate_privacy_utility(
    aggregate_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    if aggregate_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    series_specs = [
        ("event", "Event-level Gaussian DP", "o", "tab:red"),
        ("restaurant", "Restaurant-level Gaussian DP", "s", "tab:blue"),
    ]
    metric_specs = [
        ("RMSE_mean", "RMSE_std", "Test RMSE"),
        ("MAE_mean", "MAE_std", "Test MAE"),
    ]
    baseline_labels = {
        "public_opentable_adapter_w7_matched_ot": ("Non-private OpenTable", "--", "tab:green"),
        "public_only_adapter_w7_matched_ot": ("Public-only baseline", ":", "black"),
    }

    for ax, (mean_col, std_col, ylabel) in zip(axes, metric_specs):
        for privacy_mode, label, marker, color in series_specs:
            sub = aggregate_df[aggregate_df["privacy_mode"] == privacy_mode].sort_values("epsilon")
            if sub.empty:
                continue
            yerr = sub[std_col].fillna(0.0).to_numpy(dtype=float)
            ax.errorbar(
                sub["epsilon"],
                sub[mean_col],
                yerr=yerr,
                marker=marker,
                linewidth=2,
                capsize=3,
                color=color,
                label=label,
            )

        for run_name, (label, linestyle, color) in baseline_labels.items():
            base = baseline_df[baseline_df["run_name"] == run_name]
            if base.empty:
                continue
            mean_value = float(base[mean_col].iloc[0])
            std_value = base[std_col].iloc[0]
            ax.axhline(mean_value, linestyle=linestyle, color=color, linewidth=1.5, label=label)
            if pd.notna(std_value) and float(std_value) > 0:
                ax.axhspan(
                    mean_value - float(std_value),
                    mean_value + float(std_value),
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                )

        ax.set_xscale("log", base=2)
        ax.set_xticks(EPSILONS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Epsilon")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="best", frameon=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_baseline_paired_differences(per_seed: pd.DataFrame) -> pd.DataFrame:
    baseline_source = per_seed[per_seed["run_name"].isin(BASELINE_RUNS)].copy()
    if baseline_source.empty:
        return pd.DataFrame(
            columns=[
                "seed",
                "rmse_public_only",
                "rmse_public_opentable",
                "mae_public_only",
                "mae_public_opentable",
                "rmse_diff_opentable_minus_public_only",
                "mae_diff_opentable_minus_public_only",
                "better_model",
            ]
        )

    pivot = (
        baseline_source.pivot_table(index="seed", columns="run_name", values=["RMSE", "MAE"], aggfunc="first")
        .sort_index()
        .reset_index()
    )
    pivot.columns = [
        "seed" if col == ("seed", "") else f"{col[0].lower()}_{'public_only' if col[1] == BASELINE_RUNS[0] else 'public_opentable'}"
        for col in pivot.columns
    ]
    pivot["rmse_diff_opentable_minus_public_only"] = pivot["rmse_public_opentable"] - pivot["rmse_public_only"]
    pivot["mae_diff_opentable_minus_public_only"] = pivot["mae_public_opentable"] - pivot["mae_public_only"]
    pivot["better_model"] = np.where(
        pivot["rmse_diff_opentable_minus_public_only"] < 0,
        "OpenTable",
        np.where(pivot["rmse_diff_opentable_minus_public_only"] > 0, "Public-only", "Tie"),
    )
    return pivot


def load_baseline_forecast_rows(seed_dirs: list[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed_dir in seed_dirs:
        seed_label = seed_dir.name[5:] if seed_dir.name.startswith("seed_") else seed_dir.name
        for run_name in BASELINE_RUNS:
            path = seed_dir / f"forecast_28d_w7_{run_name}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["seed"] = seed_label
            df["run_name"] = run_name
            rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["date", "day_idx", "truth_cases", "pred_cases", "seed", "run_name"])
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def summarize_baseline_forecasts(forecast_rows: pd.DataFrame) -> pd.DataFrame:
    if forecast_rows.empty:
        return pd.DataFrame(
            columns=["run_name", "date", "day_idx", "truth_cases", "pred_mean", "pred_std", "seed_count"]
        )

    summary = (
        forecast_rows.groupby(["run_name", "date", "day_idx"], as_index=False)
        .agg(
            truth_cases=("truth_cases", "first"),
            pred_mean=("pred_cases", "mean"),
            pred_std=("pred_cases", "std"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["run_name", "day_idx"])
    )
    return summary


def plot_baseline_ab_comparison(
    forecast_summary: pd.DataFrame,
    paired_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    if forecast_summary.empty or paired_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1))
    ax_left, ax_right = axes

    truth_df = forecast_summary.sort_values("day_idx").drop_duplicates("day_idx")
    ax_left.plot(
        truth_df["date"],
        truth_df["truth_cases"],
        color="black",
        linewidth=1.8,
        label="Truth",
    )
    for run_name in BASELINE_RUNS:
        sub = forecast_summary[forecast_summary["run_name"] == run_name].sort_values("day_idx")
        if sub.empty:
            continue
        color = BASELINE_COLORS[run_name]
        label = BASELINE_LABELS[run_name]
        pred_std = sub["pred_std"].fillna(0.0).to_numpy(dtype=float)
        pred_mean = sub["pred_mean"].to_numpy(dtype=float)
        ax_left.plot(sub["date"], pred_mean, color=color, linewidth=1.9, label=label)
        ax_left.fill_between(
            sub["date"],
            pred_mean - pred_std,
            pred_mean + pred_std,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    ax_left.set_title("Held-out 28-day forecast trajectory")
    ax_left.set_xlabel("Date")
    ax_left.set_ylabel("Daily cases")
    ax_left.grid(alpha=0.25)
    ax_left.legend(loc="best", frameon=False)

    paired_plot = paired_df.copy().sort_values("seed")
    x = np.arange(len(paired_plot), dtype=float)
    diffs = paired_plot["rmse_diff_opentable_minus_public_only"].to_numpy(dtype=float)
    ax_right.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    ax_right.scatter(x, diffs, s=44, color="#1f77b4", zorder=3)
    ax_right.plot(x, diffs, color="#9ecae1", linewidth=1.2, zorder=2)
    mean_diff = float(np.nanmean(diffs))
    ax_right.scatter([len(paired_plot)], [mean_diff], marker="D", s=60, color="#d62728", zorder=4, label="Mean difference")
    ax_right.set_xticks(list(x) + [len(paired_plot)])
    ax_right.set_xticklabels([str(seed) for seed in paired_plot["seed"].tolist()] + ["mean"])
    ax_right.set_title("Paired seed RMSE differences")
    ax_right.set_xlabel("Seed")
    ax_right.set_ylabel("OpenTable - Public-only RMSE")
    ax_right.grid(alpha=0.25)
    ax_right.legend(loc="best", frameon=False)

    fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def aggregate_seed_outputs(
    asof: str,
    seed_dirs: list[Path],
    aggregate_dir: Path,
    experiment_tag: str,
    final_visualization_dir: Path | None = None,
    final_visualization_png_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_frames: list[pd.DataFrame] = []
    for seed_dir in seed_dirs:
        df = pd.DataFrame(build_summary_rows(asof, seed_dir))
        if df.empty:
            print(f"[warn] No matched DP metrics found in {seed_dir}")
            continue
        if "seed" not in df.columns or df["seed"].isna().all():
            df["seed"] = seed_dir.name[5:] if seed_dir.name.startswith("seed_") else seed_dir.name
        df["seed_dir"] = str(seed_dir)
        per_seed_frames.append(df)

    if not per_seed_frames:
        raise FileNotFoundError("No per-seed matched DP metrics were found for aggregation.")

    per_seed = pd.concat(per_seed_frames, ignore_index=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = aggregate_dir / "metrics_summary_dp_w7_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    dp = per_seed[
        per_seed["privacy_mode"].isin(["event", "restaurant"])
        & per_seed["epsilon"].isin([float(eps) for eps in EPSILONS] + EPSILONS)
    ].copy()
    if dp.empty:
        aggregate = pd.DataFrame(
            columns=[
                "privacy_mode",
                "mechanism",
                "epsilon",
                "seed_count",
                "RMSE_mean",
                "RMSE_std",
                "MAE_mean",
                "MAE_std",
                "MAPE_mean",
                "MAPE_std",
                "sigma_pp_mean",
                "sigma_pp_std",
            ]
        )
    else:
        aggregate = (
            dp.groupby(["privacy_mode", "mechanism", "epsilon"], as_index=False)
            .agg(
                seed_count=("seed", "nunique"),
                RMSE_mean=("RMSE", "mean"),
                RMSE_std=("RMSE", "std"),
                MAE_mean=("MAE", "mean"),
                MAE_std=("MAE", "std"),
                MAPE_mean=("MAPE", "mean"),
                MAPE_std=("MAPE", "std"),
                sigma_pp_mean=("sigma_pp", "mean"),
                sigma_pp_std=("sigma_pp", "std"),
            )
            .sort_values(["privacy_mode", "epsilon"])
        )
    aggregate_path = aggregate_dir / "metrics_summary_dp_w7_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    baseline_source = per_seed[per_seed["run_name"].isin(BASELINE_RUNS)].copy()
    baseline = (
        baseline_source.groupby(["run_name"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
        )
        if not baseline_source.empty
        else pd.DataFrame(columns=["run_name", "seed_count", "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std"])
    )
    baseline_path = aggregate_dir / "metrics_summary_dp_w7_baselines_aggregate.csv"
    baseline.to_csv(baseline_path, index=False)

    paired = build_baseline_paired_differences(per_seed)
    paired_path = aggregate_dir / "baseline_paired_differences_w7_matched_ot.csv"
    paired.to_csv(paired_path, index=False)

    forecast_rows = load_baseline_forecast_rows(seed_dirs)
    forecast_summary = summarize_baseline_forecasts(forecast_rows)
    forecast_summary_path = aggregate_dir / "baseline_forecast_test_trajectories_w7_matched_ot.csv"
    forecast_summary.to_csv(forecast_summary_path, index=False)

    plot_title = f"NYC Matched OpenTable DP Utility Across Seeds ({experiment_tag})"
    if not aggregate.empty:
        plot_aggregate_privacy_utility(
            aggregate,
            baseline,
            aggregate_dir / "privacy_utility_aggregate_rmse_mae_w7_matched_ot.png",
            plot_title,
        )

    baseline_plot_title = f"NYC OpenTable A/B baseline comparison across seeds ({experiment_tag})"
    baseline_plot_path = aggregate_dir / "baseline_ab_forecast_paired_rmse_w7_matched_ot.png"
    plot_baseline_ab_comparison(
        forecast_summary,
        paired,
        baseline_plot_path,
        baseline_plot_title,
    )

    tag = safe_name(experiment_tag)
    if final_visualization_dir is not None:
        final_visualization_dir.mkdir(parents=True, exist_ok=True)
        per_seed.to_csv(final_visualization_dir / f"{tag}_metrics_summary_dp_w7_per_seed.csv", index=False)
        aggregate.to_csv(final_visualization_dir / f"{tag}_metrics_summary_dp_w7_aggregate.csv", index=False)
        baseline.to_csv(final_visualization_dir / f"{tag}_metrics_summary_dp_w7_baselines_aggregate.csv", index=False)
        paired.to_csv(final_visualization_dir / f"{tag}_baseline_paired_differences_w7_matched_ot.csv", index=False)
        forecast_summary.to_csv(
            final_visualization_dir / f"{tag}_baseline_forecast_test_trajectories_w7_matched_ot.csv",
            index=False,
        )
        if not aggregate.empty:
            plot_aggregate_privacy_utility(
                aggregate,
                baseline,
                final_visualization_dir / f"fig6_privacy_utility_tradeoff_curves_{tag}_aggregate.pdf",
                plot_title,
            )
        plot_baseline_ab_comparison(
            forecast_summary,
            paired,
            final_visualization_dir / f"fig_baseline_ab_forecast_paired_rmse_{tag}.pdf",
            baseline_plot_title,
        )
    if final_visualization_png_dir is not None:
        final_visualization_png_dir.mkdir(parents=True, exist_ok=True)
        if not aggregate.empty:
            plot_aggregate_privacy_utility(
                aggregate,
                baseline,
                final_visualization_png_dir / f"fig6_privacy_utility_tradeoff_curves_{tag}_aggregate.png",
                plot_title,
            )
        plot_baseline_ab_comparison(
            forecast_summary,
            paired,
            final_visualization_png_dir / f"fig_baseline_ab_forecast_paired_rmse_{tag}.png",
            baseline_plot_title,
        )

    print("Saved per-seed DP summary:", per_seed_path)
    print("Saved aggregate DP summary:", aggregate_path)
    print("Saved aggregate baseline summary:", baseline_path)
    print("Saved paired baseline differences:", paired_path)
    return per_seed, aggregate


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
    ap.add_argument("--output_dir", default=None, help="Single-seed output directory. Defaults to outputs/nyc/<ASOF>.")
    ap.add_argument(
        "--run-group-dir",
        default=None,
        help="Run-group directory containing seed_<N>/ subdirectories to aggregate.",
    )
    ap.add_argument(
        "--seeds",
        default=None,
        help="Optional comma- or space-separated seed list used to choose seed_<N>/ directories.",
    )
    ap.add_argument("--experiment-tag", default="dp_multiseed", help="Experiment tag used in final visualization filenames.")
    ap.add_argument(
        "--final-visualization-dir",
        default="thesis_visualizations",
        help="Directory receiving aggregate PDF/tables for thesis-facing visualization outputs.",
    )
    ap.add_argument(
        "--final-visualization-png-dir",
        default="thesis_visualizations_png",
        help="Directory receiving aggregate PNG visualization outputs.",
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    private_dir = Path(cfg["nyc"]["paths"]["private_dir"])
    if args.run_group_dir:
        run_group_dir = Path(args.run_group_dir)
        if args.seeds:
            seed_values = [s for s in re.split(r"[\s,]+", args.seeds.strip()) if s]
            seed_dirs = [run_group_dir / f"seed_{seed}" for seed in seed_values]
        else:
            seed_dirs = sorted(path for path in run_group_dir.glob("seed_*") if path.is_dir())
        aggregate_seed_outputs(
            asof=args.asof,
            seed_dirs=seed_dirs,
            aggregate_dir=run_group_dir / "aggregate",
            experiment_tag=args.experiment_tag,
            final_visualization_dir=Path(args.final_visualization_dir) if args.final_visualization_dir else None,
            final_visualization_png_dir=Path(args.final_visualization_png_dir)
            if args.final_visualization_png_dir
            else None,
        )
        return

    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "nyc" / args.asof
    write_single_seed_summary(args.asof, out_dir, private_dir)
    print("Saved DP metrics summary:", out_dir / "metrics_summary_dp_w7.csv")


if __name__ == "__main__":
    main()

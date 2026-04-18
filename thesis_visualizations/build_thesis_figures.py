import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "thesis_visualizations"
FIG_EXT = "pdf"
FIG_DPI = 300
ONLINE_DIR = ROOT / "data" / "processed" / "online"
PRIVATE_DIR = ROOT / "data" / "processed" / "private"
OUT_DIR = ROOT / "outputs" / "nyc" / "2020-08-05"


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> dict:
    with open(ensure_exists(path), "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(run_name: str) -> dict:
    return load_json(OUT_DIR / f"metrics_{run_name}.json")


def figure_path(stem: str) -> Path:
    return FIG_DIR / f"{stem}.{FIG_EXT}"


def save_figure(fig, stem: str) -> Path:
    out_path = figure_path(stem)
    save_kwargs = {"bbox_inches": "tight"}
    if FIG_EXT.lower() != "pdf":
        save_kwargs["dpi"] = FIG_DPI
    fig.savefig(out_path, **save_kwargs)
    return out_path


def fig2_timeline() -> tuple[Path, list[str]]:
    split_w7 = load_json(ONLINE_DIR / "split_info_2020-08-05_matched_ot_w7.json")
    split_w0 = load_json(ONLINE_DIR / "split_info_2020-08-05_matched_ot_w0.json")
    split_w3 = load_json(ONLINE_DIR / "split_info_2020-08-05_matched_ot_w3.json")

    public_start = pd.to_datetime(split_w7["public_start"])
    public_end = pd.to_datetime(split_w7["public_end"])
    ot_start = pd.to_datetime(split_w7["opentable_observed_start"])
    ot_end = pd.to_datetime(split_w7["opentable_observed_end"])
    joint_start = pd.to_datetime(split_w7["window_start"])
    joint_end = pd.to_datetime(split_w7["window_end"])
    train_start = pd.to_datetime(split_w7["train_start"])
    train_end = pd.to_datetime(split_w7["train_end"])
    test_start = pd.to_datetime(split_w7["test_start"])
    test_end = pd.to_datetime(split_w7["test_end"])

    fig, ax = plt.subplots(figsize=(11.0, 4.0))

    y_public, y_ot, y_contract = 2.2, 1.2, 0.2
    ax.hlines(y_public, public_start, public_end, color="#666666", linewidth=8, label="Public data coverage")
    ax.hlines(y_ot, ot_start, ot_end, color="#1f77b4", linewidth=8, label="OpenTable observed coverage")
    ax.hlines(y_contract, joint_start, joint_end, color="#c7c7c7", linewidth=10, label="Canonical matched window")
    ax.hlines(y_contract, train_start, train_end, color="#2ca02c", linewidth=10, label="Train interval")
    ax.hlines(y_contract, test_start, test_end, color="#d62728", linewidth=10, label="Held-out 28-day test interval")

    for x, text, y, dx_days, dy, ha, valign in [
        (joint_start, "Matched start\n2020-02-29", y_contract, -8, -0.46, "center", "top"),
        (joint_end, "Matched end\n2020-08-05", y_contract, 18, 0.54, "center", "bottom"),
    ]:
        ax.vlines(x, y - 0.24, y + 0.24, color="black", linewidth=0.9)
        ax.text(x + pd.Timedelta(days=dx_days), y + dy, text, ha=ha, va=valign)

    ax.vlines(train_end, y_contract - 0.24, y_contract + 0.24, color="black", linewidth=0.9)
    ax.vlines(test_start, y_contract - 0.24, y_contract + 0.24, color="black", linewidth=0.9)
    boundary_x = train_end + pd.Timedelta(days=1.5)
    ax.text(boundary_x, y_contract - 0.44, "Train/Test boundary", ha="center", va="top")

    ax.text(public_start, y_public + 0.22, "Public data", ha="left", va="bottom")
    ax.text(ot_start, y_ot + 0.22, "OpenTable", ha="left", va="bottom")
    ax.text(joint_start, y_contract + 0.28, "Matched-window contract", ha="left", va="bottom")
    ax.text(pd.Timestamp("2020-12-15"), y_public + 0.22, "Public coverage continues\nthrough 2022-10-15", ha="right", va="bottom")
    ax.text(
        test_start + (test_end - test_start) / 2 - pd.Timedelta(days=10),
        y_contract + 0.06,
        "Held-out 28-day\ntest horizon",
        ha="center",
        va="bottom",
    )

    ax.set_title("Canonical Matched-Window Timeline")
    ax.set_xlabel("Date")
    ax.set_yticks([])
    ax.set_ylim(-0.8, 2.8)
    ax.set_xlim(public_start - pd.Timedelta(days=12), pd.Timestamp("2021-01-01"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    legend_handles = [
        Line2D([0], [0], color="#666666", lw=8, label="Public data coverage"),
        Line2D([0], [0], color="#1f77b4", lw=8, label="OpenTable observed coverage"),
        Patch(facecolor="#2ca02c", label="Train interval"),
        Patch(facecolor="#d62728", label="Held-out test interval"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)
    fig.tight_layout()

    out_path = save_figure(fig, "fig2_matched_window_timeline")
    plt.close(fig)

    used = [
        str((ONLINE_DIR / "split_info_2020-08-05_matched_ot_w7.json").relative_to(ROOT)),
        str((ONLINE_DIR / "split_info_2020-08-05_matched_ot_w0.json").relative_to(ROOT)),
        str((ONLINE_DIR / "split_info_2020-08-05_matched_ot_w3.json").relative_to(ROOT)),
    ]
    return out_path, used


def plot_fit_panel(ax, run_name: str, panel_title: str) -> None:
    df = pd.read_csv(ensure_exists(OUT_DIR / f"fit_train_test_{run_name}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    ax.plot(df["date"], df["truth_cases"], color="black", linewidth=1.4, label="Truth")
    ax.plot(df["date"], df["pred_cases"], color="#1f77b4", linewidth=1.4, label="Prediction")
    if test_mask.any():
        test_start = df.loc[test_mask, "date"].iloc[0]
        ax.axvspan(test_start, df["date"].iloc[-1], color="#f0f0f0", alpha=1.0, zorder=-2)
        ax.axvline(test_start, color="#b22222", linestyle="--", linewidth=1.0)
    ax.set_title(panel_title)
    ax.grid(color="#e5e5e5", linewidth=0.6)


def fig3_ab_forecasts() -> tuple[Path, list[str]]:
    runs = {
        0: ("public_only_adapter_w0_matched_ot", "public_opentable_adapter_w0_matched_ot"),
        3: ("public_only_adapter_w3_matched_ot", "public_opentable_adapter_w3_matched_ot"),
        7: ("public_only_adapter_w7_matched_ot", "public_opentable_adapter_w7_matched_ot"),
    }
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.4), sharex=True)
    used = []
    for row_idx, w in enumerate([0, 3, 7]):
        left_run, right_run = runs[w]
        plot_fit_panel(axes[row_idx, 0], left_run, f"Public-only, w={w}")
        plot_fit_panel(axes[row_idx, 1], right_run, f"OpenTable, w={w}")
        used.extend(
            [
                str((OUT_DIR / f"fit_train_test_{left_run}.csv").relative_to(ROOT)),
                str((OUT_DIR / f"fit_train_test_{right_run}.csv").relative_to(ROOT)),
            ]
        )

    axes[0, 0].set_ylabel("Daily cases")
    axes[1, 0].set_ylabel("Daily cases")
    axes[2, 0].set_ylabel("Daily cases")
    axes[2, 0].set_xlabel("Date")
    axes[2, 1].set_xlabel("Date")

    handles = [
        Line2D([0], [0], color="black", lw=1.4, label="Truth"),
        Line2D([0], [0], color="#1f77b4", lw=1.4, label="Prediction"),
        Patch(facecolor="#f0f0f0", edgecolor="none", label="Held-out test segment"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle("Matched-Window Forecast Comparison Across Smoothing Regimes", y=0.992)
    fig.tight_layout(rect=[0, 0, 1, 0.935])

    out_path = save_figure(fig, "fig3_ab_forecast_comparison_w0_w3_w7")
    plt.close(fig)
    return out_path, used


def fig4_smoothing_summary() -> tuple[Path, list[str]]:
    summary_path = ensure_exists(OUT_DIR / "metrics_summary.csv")
    df = pd.read_csv(summary_path)
    subset = df[df["run_tag"].isin(
        [
            "public_only_adapter_w0_matched_ot",
            "public_only_adapter_w3_matched_ot",
            "public_only_adapter_w7_matched_ot",
            "public_opentable_adapter_w0_matched_ot",
            "public_opentable_adapter_w3_matched_ot",
            "public_opentable_adapter_w7_matched_ot",
        ]
    )].copy()

    subset["condition"] = subset["run_tag"].map(
        lambda x: "Public-only" if x.startswith("public_only") else "OpenTable"
    )
    subset["smooth_cases_window"] = subset["smooth_cases_window"].astype(int)
    subset = subset.sort_values(["condition", "smooth_cases_window"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    color_map = {"Public-only": "#555555", "OpenTable": "#1f77b4"}

    for condition, group in subset.groupby("condition"):
        axes[0].plot(
            group["smooth_cases_window"],
            group["test_rmse"],
            marker="o",
            linewidth=1.8,
            color=color_map[condition],
            label=condition,
        )
        axes[1].plot(
            group["smooth_cases_window"],
            group["test_mae"],
            marker="o",
            linewidth=1.8,
            color=color_map[condition],
            label=condition,
        )

    axes[0].set_title("Test RMSE")
    axes[1].set_title("Test MAE")
    axes[0].set_ylabel("Error")
    for ax in axes:
        ax.set_xlabel("Smoothing window w")
        ax.set_xticks([0, 3, 7])
        ax.grid(color="#e5e5e5", linewidth=0.6)
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Smoothing Sensitivity in the Matched A/B Comparison", y=1.02)
    fig.tight_layout()

    out_path = save_figure(fig, "fig4_smoothing_sensitivity_summary")
    plt.close(fig)

    used = [str(summary_path.relative_to(ROOT))]
    for run in [
        "public_only_adapter_w0_matched_ot",
        "public_only_adapter_w3_matched_ot",
        "public_only_adapter_w7_matched_ot",
        "public_opentable_adapter_w0_matched_ot",
        "public_opentable_adapter_w3_matched_ot",
        "public_opentable_adapter_w7_matched_ot",
    ]:
        used.append(str((OUT_DIR / f"metrics_{run}.json").relative_to(ROOT)))
    return out_path, used


def fig5_original_vs_dp() -> tuple[Path, list[str]]:
    base = pd.read_csv(ensure_exists(PRIVATE_DIR / "opentable_private_observed_2020-08-05_matched_ot_series.csv"))
    event = pd.read_csv(ensure_exists(PRIVATE_DIR / "opentable_private_observed_dp_gaussian_event_eps1_2020-08-05_matched_ot_series.csv"))
    rest = pd.read_csv(ensure_exists(PRIVATE_DIR / "opentable_private_observed_dp_gaussian_restaurant_eps1_2020-08-05_matched_ot_series.csv"))
    for df in (base, event, rest):
        df["date"] = pd.to_datetime(df["date"])

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.6), sharex=True)
    series_specs = [
        (base, "Original non-private OpenTable signal", "black"),
        (event, "Event-level Gaussian DP, epsilon = 1", "#d62728"),
        (rest, "Restaurant-level Gaussian DP, epsilon = 1", "#1f77b4"),
    ]
    for ax, (df, title, color) in zip(axes, series_specs):
        ax.plot(df["date"], df["released_yoy_pp"], color=color, linewidth=1.5)
        ax.set_title(title)
        ax.set_ylabel("YoY seated-diner signal (pp)")
        ax.grid(color="#e5e5e5", linewidth=0.6)
    axes[-1].set_xlabel("Date")
    fig.suptitle("OpenTable Signal Before and After Gaussian DP", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = save_figure(fig, "fig5_opentable_original_vs_dp_noised")
    plt.close(fig)
    used = [
        str((PRIVATE_DIR / "opentable_private_observed_2020-08-05_matched_ot_series.csv").relative_to(ROOT)),
        str((PRIVATE_DIR / "opentable_private_observed_dp_gaussian_event_eps1_2020-08-05_matched_ot_series.csv").relative_to(ROOT)),
        str((PRIVATE_DIR / "opentable_private_observed_dp_gaussian_restaurant_eps1_2020-08-05_matched_ot_series.csv").relative_to(ROOT)),
    ]
    return out_path, used


def fig6_privacy_utility() -> tuple[Path, list[str]]:
    event_runs = [
        "public_opentable_dp_gaussian_event_eps1_w7_matched_ot",
        "public_opentable_dp_gaussian_event_eps2_w7_matched_ot",
        "public_opentable_dp_gaussian_event_eps4_w7_matched_ot",
        "public_opentable_dp_gaussian_event_eps8_w7_matched_ot",
        "public_opentable_dp_gaussian_event_eps16_w7_matched_ot",
    ]
    rest_runs = [
        "public_opentable_dp_gaussian_restaurant_eps1_w7_matched_ot",
        "public_opentable_dp_gaussian_restaurant_eps2_w7_matched_ot",
        "public_opentable_dp_gaussian_restaurant_eps4_w7_matched_ot",
        "public_opentable_dp_gaussian_restaurant_eps8_w7_matched_ot",
        "public_opentable_dp_gaussian_restaurant_eps16_w7_matched_ot",
    ]
    event_rows = []
    rest_rows = []
    used = []
    for run in event_runs:
        m = load_metrics(run)
        event_rows.append((float(m["epsilon"]), float(m["test_metrics"]["rmse"])))
        used.append(str((OUT_DIR / f"metrics_{run}.json").relative_to(ROOT)))
    for run in rest_runs:
        m = load_metrics(run)
        rest_rows.append((float(m["epsilon"]), float(m["test_metrics"]["rmse"])))
        used.append(str((OUT_DIR / f"metrics_{run}.json").relative_to(ROOT)))

    public_only = load_metrics("public_only_adapter_w7_matched_ot")
    nonprivate = load_metrics("public_opentable_adapter_w7_matched_ot")
    used.extend(
        [
            str((OUT_DIR / "metrics_public_only_adapter_w7_matched_ot.json").relative_to(ROOT)),
            str((OUT_DIR / "metrics_public_opentable_adapter_w7_matched_ot.json").relative_to(ROOT)),
        ]
    )

    event_df = pd.DataFrame(event_rows, columns=["epsilon", "rmse"]).sort_values("epsilon")
    rest_df = pd.DataFrame(rest_rows, columns=["epsilon", "rmse"]).sort_values("epsilon")

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.plot(event_df["epsilon"], event_df["rmse"], marker="o", linewidth=1.9, color="#d62728", label="Event-level Gaussian DP")
    ax.plot(rest_df["epsilon"], rest_df["rmse"], marker="s", linewidth=1.9, color="#1f77b4", label="Restaurant-level Gaussian DP")
    ax.axhline(float(nonprivate["test_metrics"]["rmse"]), color="#2ca02c", linestyle="--", linewidth=1.4, label="Non-private OpenTable baseline")
    ax.axhline(float(public_only["test_metrics"]["rmse"]), color="#444444", linestyle=":", linewidth=1.4, label="Public-only baseline")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Privacy-Utility Tradeoff Across the Gaussian DP Sweep")
    ax.grid(color="#e5e5e5", linewidth=0.6)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    out_path = save_figure(fig, "fig6_privacy_utility_tradeoff_curves")
    plt.close(fig)
    return out_path, used


def main() -> None:
    global FIG_DIR, FIG_EXT, FIG_DPI

    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"], help="Output figure format.")
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Directory for rendered figures. Defaults to thesis_visualizations for pdf and thesis_visualizations_png for png.",
    )
    ap.add_argument("--dpi", type=int, default=300, help="Raster DPI used for non-PDF outputs.")
    args = ap.parse_args()

    FIG_EXT = args.format
    FIG_DPI = int(args.dpi)
    if args.output_dir:
        FIG_DIR = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    elif FIG_EXT == "png":
        FIG_DIR = ROOT / "thesis_visualizations_png"

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    created = []
    summary = {}
    for builder in [fig2_timeline, fig3_ab_forecasts, fig4_smoothing_summary, fig5_original_vs_dp, fig6_privacy_utility]:
        out_path, used = builder()
        created.append(out_path)
        summary[str(out_path.relative_to(ROOT))] = used

    print("Created figure files:")
    for path in created:
        print(f"- {path.relative_to(ROOT)}")
    print("\nSource artifacts used:")
    for fig, used in summary.items():
        print(f"- {fig}")
        for src in used:
            print(f"  - {src}")
    print("\nWorkarounds:")
    print("- None. All required source artifacts were present in the workspace.")


if __name__ == "__main__":
    main()

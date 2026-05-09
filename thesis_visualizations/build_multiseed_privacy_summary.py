import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".venv/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_GROUP = ROOT / "outputs" / "nyc" / "2020-08-05" / "run_groups" / "dp_multiseed_20260417_223607"
PDF_DIR = ROOT / "thesis_visualizations"
PNG_DIR = ROOT / "thesis_visualizations_png"


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

EPSILONS = [1, 2, 4, 8, 16]


def save_figure(fig, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"}
    if out_path.suffix.lower() != ".pdf":
        kwargs["dpi"] = int(dpi)
    fig.savefig(out_path, **kwargs)


def load_aggregate_tables(run_group_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = pd.read_csv(run_group_dir / "aggregate" / "metrics_summary_dp_w7_aggregate.csv")
    baselines = pd.read_csv(run_group_dir / "aggregate" / "metrics_summary_dp_w7_baselines_aggregate.csv")
    return aggregate, baselines


def build_figure(aggregate: pd.DataFrame, baselines: pd.DataFrame, out_path: Path, dpi: int) -> None:
    event_df = aggregate[aggregate["privacy_mode"] == "event"].sort_values("epsilon")
    rest_df = aggregate[aggregate["privacy_mode"] == "restaurant"].sort_values("epsilon")
    nonprivate = baselines[baselines["run_name"] == "public_opentable_adapter_w7_matched_ot"]
    public_only = baselines[baselines["run_name"] == "public_only_adapter_w7_matched_ot"]

    fig, ax = plt.subplots(figsize=(9.4, 5.2))

    if not event_df.empty:
        ax.plot(
            event_df["epsilon"],
            event_df["RMSE_mean"],
            marker="o",
            linewidth=1.9,
            color="#d62728",
            label="Event-level Gaussian DP",
        )
        ax.fill_between(
            event_df["epsilon"],
            event_df["RMSE_mean"] - event_df["RMSE_std"].fillna(0.0),
            event_df["RMSE_mean"] + event_df["RMSE_std"].fillna(0.0),
            color="#d62728",
            alpha=0.16,
            linewidth=0,
        )

    if not rest_df.empty:
        ax.plot(
            rest_df["epsilon"],
            rest_df["RMSE_mean"],
            marker="s",
            linewidth=1.9,
            color="#1f77b4",
            label="Restaurant-level Gaussian DP",
        )
        ax.fill_between(
            rest_df["epsilon"],
            rest_df["RMSE_mean"] - rest_df["RMSE_std"].fillna(0.0),
            rest_df["RMSE_mean"] + rest_df["RMSE_std"].fillna(0.0),
            color="#1f77b4",
            alpha=0.16,
            linewidth=0,
        )

    if not nonprivate.empty:
        mean = float(nonprivate["RMSE_mean"].iloc[0])
        std = float(nonprivate["RMSE_std"].fillna(0.0).iloc[0])
        ax.axhline(mean, color="#2ca02c", linestyle="--", linewidth=1.4, label="Non-private OpenTable baseline")
        if std > 0:
            ax.axhspan(mean - std, mean + std, color="#2ca02c", alpha=0.10, linewidth=0)

    if not public_only.empty:
        mean = float(public_only["RMSE_mean"].iloc[0])
        std = float(public_only["RMSE_std"].fillna(0.0).iloc[0])
        ax.axhline(mean, color="#444444", linestyle=":", linewidth=1.4, label="Public-only baseline")
        if std > 0:
            ax.axhspan(mean - std, mean + std, color="#444444", alpha=0.08, linewidth=0)

    ax.set_xscale("log", base=2)
    ax.set_xticks(EPSILONS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Test RMSE (mean across 5 seeds)")
    ax.set_title("Privacy-Utility Tradeoff Across the Gaussian DP Sweep")
    ax.grid(color="#e5e5e5", linewidth=0.6)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    save_figure(fig, out_path, dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-group-dir", default=str(DEFAULT_RUN_GROUP), help="Run-group directory with aggregate CSVs.")
    parser.add_argument(
        "--stem",
        default="fig6_privacy_utility_tradeoff_curves_multiseed",
        help="Output filename stem for PDF and PNG.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG output DPI.")
    args = parser.parse_args()

    run_group_dir = Path(args.run_group_dir)
    aggregate, baselines = load_aggregate_tables(run_group_dir)
    pdf_path = PDF_DIR / f"{args.stem}.pdf"
    png_path = PNG_DIR / f"{args.stem}.png"
    build_figure(aggregate, baselines, pdf_path, args.dpi)
    build_figure(aggregate, baselines, png_path, args.dpi)

    print("Created multiseed privacy summary figure:")
    print(f"- {pdf_path.relative_to(ROOT)}")
    print(f"- {png_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

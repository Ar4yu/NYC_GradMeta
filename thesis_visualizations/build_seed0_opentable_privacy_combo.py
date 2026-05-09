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
DEFAULT_BASELINE_OUTPUT_DIR = ROOT / "outputs" / "nyc" / "2020-08-05"
DEFAULT_PRIVACY_RUN_GROUP = ROOT / "outputs" / "nyc" / "2020-08-05" / "run_groups" / "dp_multiseed_20260417_223607"
PDF_DIR = ROOT / "thesis_visualizations"
PNG_DIR = ROOT / "thesis_visualizations_png"
EPSILONS = [1, 2, 4, 8, 16]
PUBLIC_ONLY_RUN = "public_only_adapter_w7_matched_ot"
OPENTABLE_RUN = "public_opentable_adapter_w7_matched_ot"


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


def save_figure(fig, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"}
    if out_path.suffix.lower() != ".pdf":
        kwargs["dpi"] = int(dpi)
    fig.savefig(out_path, **kwargs)


def load_original_baselines(output_dir: Path) -> tuple[float, float]:
    metrics = pd.read_csv(output_dir / "metrics_summary.csv")
    public_only = metrics[metrics["run_tag"] == PUBLIC_ONLY_RUN]
    opentable = metrics[metrics["run_tag"] == OPENTABLE_RUN]
    if public_only.empty or opentable.empty:
        raise ValueError("Missing original long-train baseline rows in metrics_summary.csv")
    return float(public_only["test_rmse"].iloc[0]), float(opentable["test_rmse"].iloc[0])


def load_privacy_aggregate(run_group_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_group_dir / "aggregate" / "metrics_summary_dp_w7_aggregate.csv")


def build_privacy_figure(
    aggregate: pd.DataFrame,
    public_only_rmse: float,
    opentable_rmse: float,
    out_path: Path,
    dpi: int,
    show_std: bool,
) -> None:
    event_df = aggregate[aggregate["privacy_mode"] == "event"].sort_values("epsilon")
    restaurant_df = aggregate[aggregate["privacy_mode"] == "restaurant"].sort_values("epsilon")

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
        if show_std:
            ax.fill_between(
                event_df["epsilon"],
                event_df["RMSE_mean"] - event_df["RMSE_std"].fillna(0.0),
                event_df["RMSE_mean"] + event_df["RMSE_std"].fillna(0.0),
                color="#d62728",
                alpha=0.16,
                linewidth=0,
            )

    if not restaurant_df.empty:
        ax.plot(
            restaurant_df["epsilon"],
            restaurant_df["RMSE_mean"],
            marker="s",
            linewidth=1.9,
            color="#1f77b4",
            label="Restaurant-level Gaussian DP",
        )
        if show_std:
            ax.fill_between(
                restaurant_df["epsilon"],
                restaurant_df["RMSE_mean"] - restaurant_df["RMSE_std"].fillna(0.0),
                restaurant_df["RMSE_mean"] + restaurant_df["RMSE_std"].fillna(0.0),
                color="#1f77b4",
                alpha=0.16,
                linewidth=0,
            )

    ax.axhline(opentable_rmse, color="#2ca02c", linestyle="--", linewidth=1.4, label="Non-private OpenTable")
    ax.axhline(public_only_rmse, color="#444444", linestyle=":", linewidth=1.4, label="Public-only baseline")

    ax.set_xscale("log", base=2)
    ax.set_xticks(EPSILONS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Privacy-Utility Tradeoff Across the Gaussian DP Sweep")
    ax.grid(color="#e5e5e5", linewidth=0.6)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    save_figure(fig, out_path, dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-output-dir", default=str(DEFAULT_BASELINE_OUTPUT_DIR))
    parser.add_argument("--privacy-run-group", default=str(DEFAULT_PRIVACY_RUN_GROUP))
    parser.add_argument("--stem-with-std", default="fig_privacy_utility_tradeoff_original_long_train_with_std")
    parser.add_argument("--stem-no-std", default="fig_privacy_utility_tradeoff_original_long_train_no_std")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    baseline_output_dir = Path(args.baseline_output_dir)
    privacy_run_group = Path(args.privacy_run_group)
    public_only_rmse, opentable_rmse = load_original_baselines(baseline_output_dir)
    aggregate = load_privacy_aggregate(privacy_run_group)

    outputs = [
        (args.stem_with_std, True),
        (args.stem_no_std, False),
    ]

    created: list[Path] = []
    for stem, show_std in outputs:
        pdf_path = PDF_DIR / f"{stem}.pdf"
        png_path = PNG_DIR / f"{stem}.png"
        build_privacy_figure(aggregate, public_only_rmse, opentable_rmse, pdf_path, args.dpi, show_std)
        build_privacy_figure(aggregate, public_only_rmse, opentable_rmse, png_path, args.dpi, show_std)
        created.extend([pdf_path, png_path])

    print("Created figures:")
    for path in created:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

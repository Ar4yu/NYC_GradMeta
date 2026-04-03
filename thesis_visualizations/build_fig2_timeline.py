import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "thesis_visualizations"
ONLINE_DIR = ROOT / "data" / "processed" / "online"


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_band(ax, start, end, y, height, color, alpha=1.0, zorder=2):
    x0 = mdates.date2num(pd.Timestamp(start))
    x1 = mdates.date2num(pd.Timestamp(end))
    rect = Rectangle((x0, y - height / 2), x1 - x0, height, facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder)
    ax.add_patch(rect)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    w7_path = ONLINE_DIR / "split_info_2020-08-05_matched_ot_w7.json"
    w0_path = ONLINE_DIR / "split_info_2020-08-05_matched_ot_w0.json"
    w3_path = ONLINE_DIR / "split_info_2020-08-05_matched_ot_w3.json"

    info = load_json(w7_path)
    info_w0 = load_json(w0_path)
    info_w3 = load_json(w3_path)

    # Consistency check across smoothing windows.
    for key in ("window_start", "window_end", "train_start", "train_end", "test_start", "test_end"):
        if not (info[key] == info_w0[key] == info_w3[key]):
            raise ValueError(f"Inconsistent matched-window metadata for key '{key}'.")

    matched_start = pd.Timestamp(info["window_start"])
    matched_end = pd.Timestamp(info["window_end"])
    train_start = pd.Timestamp(info["train_start"])
    train_end = pd.Timestamp(info["train_end"])
    test_start = pd.Timestamp(info["test_start"])
    test_end = pd.Timestamp(info["test_end"])
    public_start = pd.Timestamp(info["public_start"])
    opentable_start = pd.Timestamp(info["opentable_observed_start"])

    x_min = pd.Timestamp("2020-02-15")
    x_max = pd.Timestamp("2020-08-24")

    fig, ax = plt.subplots(figsize=(10.2, 3.8))

    y_public = 2.35
    y_ot = 1.45
    y_contract = 0.55
    band_h = 0.16

    # Public coverage within visible range plus subtle continuation arrow.
    add_band(ax, max(public_start, x_min), x_max - pd.Timedelta(days=10), y_public, band_h, "#7a7a7a")
    ax.annotate(
        "",
        xy=(x_max - pd.Timedelta(days=1), y_public),
        xytext=(x_max - pd.Timedelta(days=11), y_public),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#7a7a7a"),
        zorder=3,
    )
    ax.text(matched_start, y_public + 0.18, "Public data coverage", ha="left", va="bottom")
    ax.text(x_max - pd.Timedelta(days=2), y_public + 0.18, "continues beyond\nvisible range", ha="right", va="bottom", color="#555555")

    # OpenTable observed coverage.
    add_band(ax, max(opentable_start, x_min), matched_end, y_ot, band_h, "#2c7fb8")
    ax.text(matched_start, y_ot + 0.18, "OpenTable observed coverage", ha="left", va="bottom")

    # Canonical matched-window contract.
    add_band(ax, matched_start, matched_end, y_contract, band_h, "#d9d9d9", alpha=1.0, zorder=1)
    add_band(ax, train_start, train_end, y_contract, band_h, "#2ca02c", zorder=3)
    add_band(ax, test_start, test_end, y_contract, band_h, "#d62728", zorder=4)
    ax.text(matched_start, y_contract + 0.18, "Canonical matched-window contract", ha="left", va="bottom")

    # Boundary marker.
    ax.vlines([train_end, test_start], y_contract - 0.22, y_contract + 0.22, colors="black", linewidth=0.9, zorder=5)

    # Annotations placed in separate zones to avoid collisions.
    ax.annotate(
        "Matched start\n2020-02-29",
        xy=(matched_start, y_contract),
        xytext=(matched_start - pd.Timedelta(days=4), y_contract - 0.35),
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="-", lw=0.9, color="black"),
    )
    ax.annotate(
        "Train/Test boundary",
        xy=(test_start, y_contract),
        xytext=(test_start, y_contract - 0.36),
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="-", lw=0.9, color="black"),
    )
    ax.annotate(
        "Held-out 28-day\nforecast horizon",
        xy=(test_start + (test_end - test_start) / 2, y_contract),
        xytext=(test_start + pd.Timedelta(days=8), y_contract + 0.26),
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="-", lw=0.9, color="black"),
    )
    ax.annotate(
        "Matched end\n2020-08-05",
        xy=(matched_end, y_contract),
        xytext=(matched_end + pd.Timedelta(days=10), y_contract + 0.52),
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="-", lw=0.9, color="black"),
    )

    ax.set_title("Canonical Matched-Window Timeline")
    ax.set_xlabel("Date")
    ax.set_yticks([])
    ax.set_ylim(-0.15, 2.85)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="x", color="#e3e3e3", linewidth=0.6)

    fig.tight_layout()
    out_path = FIG_DIR / "fig2_matched_window_timeline.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print("Output file created:")
    print(f"- {out_path.relative_to(ROOT)}")
    print("Metadata files used:")
    print(f"- {w7_path.relative_to(ROOT)}")
    print(f"- {w0_path.relative_to(ROOT)}")
    print(f"- {w3_path.relative_to(ROOT)}")
    print("Rendered contract dates:")
    print(f"- matched start: {matched_start.date()}")
    print(f"- matched end: {matched_end.date()}")
    print(f"- train: {train_start.date()} to {train_end.date()}")
    print(f"- test: {test_start.date()} to {test_end.date()}")


if __name__ == "__main__":
    main()

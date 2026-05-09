import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".venv/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "data" / "processed" / "contact_matrix_us.csv"
PDF_DIR = ROOT / "thesis_visualizations"
PNG_DIR = ROOT / "thesis_visualizations_png"


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def pretty_age_label(label: object) -> str:
    text = str(label).strip()
    if text.endswith("+"):
        return text.lstrip("0")
    if "-" not in text:
        return text
    left, right = text.split("-", 1)
    lo = int(left)
    hi = int(right) - 1
    return f"{lo}-{hi}"


def load_row_normalized_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square contact matrix, got {matrix.shape}.")

    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Contact matrix contains non-finite values: {path}")

    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(f"Contact matrix contains a non-positive row sum: {path}")

    normalized = values / row_sums
    labels = [pretty_age_label(label) for label in matrix.index]
    return pd.DataFrame(normalized, index=labels, columns=labels)


def build_heatmap(matrix: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", aspect="equal")

    tick_positions = np.arange(matrix.shape[0])
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Contact age group")
    ax.set_ylabel("Source age group")
    ax.set_title("Age-Structured Contact Matrix Used by the Simulator")

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized contact share")

    fig.text(
        0.5,
        0.015,
        "Rows are normalized to sum to 1, matching the simulator preprocessing of data/processed/contact_matrix_us.csv.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight"}
    if out_path.suffix.lower() != ".pdf":
        save_kwargs["dpi"] = int(dpi)
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX), help="Contact matrix CSV with age bins as rows/columns.")
    parser.add_argument("--stem", default="figA1_age_contact_matrix_heatmap", help="Output filename stem.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG output DPI.")
    args = parser.parse_args()

    matrix = load_row_normalized_matrix(Path(args.matrix))
    outputs = [
        PDF_DIR / f"{args.stem}.pdf",
        PNG_DIR / f"{args.stem}.png",
    ]
    for out_path in outputs:
        build_heatmap(matrix, out_path, args.dpi)

    print("Created contact matrix heatmap:")
    for out_path in outputs:
        print(f"- {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "thesis_tables"
DEFAULT_RUN_GROUP = ROOT / "outputs" / "nyc" / "2020-08-05" / "run_groups" / "dp_multiseed_20260417_223607"


def latex_escape(value: str) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def metric_str(value) -> str:
    return f"{float(value):.4f}"


def plus_minus(mean_value, std_value) -> str:
    return rf"{metric_str(mean_value)} $\pm$ {metric_str(std_value if pd.notna(std_value) else 0.0)}"


def format_table(
    caption: str,
    label: str,
    colspec: str,
    header: list[str],
    rows: list[list[str]],
    notes: list[str] | None = None,
    size_cmd: str = r"\footnotesize",
) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        size_cmd,
        r"\begin{threeparttable}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabularx}}{{\textwidth}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        if len(row) == 1:
            lines.append(row[0] + r" \\")
        else:
            lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    if notes:
        lines.append(r"\begin{tablenotes}[flushleft]")
        lines.append(r"\footnotesize")
        for note in notes:
            lines.append(r"\item " + note)
        lines.append(r"\end{tablenotes}")
    lines.extend([r"\end{threeparttable}", r"\end{table}", ""])
    return "\n".join(lines)


def build_combined_summary(run_group_dir: Path) -> pd.DataFrame:
    aggregate = pd.read_csv(run_group_dir / "aggregate" / "metrics_summary_dp_w7_aggregate.csv")
    baselines = pd.read_csv(run_group_dir / "aggregate" / "metrics_summary_dp_w7_baselines_aggregate.csv")

    baseline_rows = []
    label_map = {
        "public_only_adapter_w7_matched_ot": "Public-only baseline",
        "public_opentable_adapter_w7_matched_ot": "Non-private OpenTable baseline",
    }
    for _, row in baselines.iterrows():
        baseline_rows.append(
            {
                "condition_group": "Matched non-private baselines",
                "condition_label": label_map.get(row["run_name"], row["run_name"]),
                "privacy_mode": "none",
                "epsilon": None,
                "seed_count": int(row["seed_count"]),
                "RMSE_mean": float(row["RMSE_mean"]),
                "RMSE_std": float(row["RMSE_std"]) if pd.notna(row["RMSE_std"]) else 0.0,
                "MAE_mean": float(row["MAE_mean"]),
                "MAE_std": float(row["MAE_std"]) if pd.notna(row["MAE_std"]) else 0.0,
            }
        )

    dp_rows = []
    for _, row in aggregate.iterrows():
        mode = str(row["privacy_mode"])
        dp_rows.append(
            {
                "condition_group": "Event-level Gaussian DP" if mode == "event" else "Restaurant-level Gaussian DP",
                "condition_label": "Event-DP OpenTable" if mode == "event" else "Restaurant-DP OpenTable",
                "privacy_mode": mode,
                "epsilon": float(row["epsilon"]),
                "seed_count": int(row["seed_count"]),
                "RMSE_mean": float(row["RMSE_mean"]),
                "RMSE_std": float(row["RMSE_std"]) if pd.notna(row["RMSE_std"]) else 0.0,
                "MAE_mean": float(row["MAE_mean"]),
                "MAE_std": float(row["MAE_std"]) if pd.notna(row["MAE_std"]) else 0.0,
                "MAPE_mean": float(row["MAPE_mean"]) if pd.notna(row["MAPE_mean"]) else None,
                "MAPE_std": float(row["MAPE_std"]) if pd.notna(row["MAPE_std"]) else 0.0,
            }
        )

    combined = pd.DataFrame(baseline_rows + dp_rows)
    return combined


def build_tex_rows(summary: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    group_order = [
        "Matched non-private baselines",
        "Event-level Gaussian DP",
        "Restaurant-level Gaussian DP",
    ]
    for group_name in group_order:
        group_df = summary[summary["condition_group"] == group_name].copy()
        if group_df.empty:
            continue
        if "epsilon" in group_df.columns:
            group_df = group_df.sort_values("epsilon", na_position="first")
        rows.append([rf"\multicolumn{{5}}{{l}}{{\textit{{{group_name}}}}}"])
        for _, row in group_df.iterrows():
            epsilon = r"\textemdash" if pd.isna(row.get("epsilon")) else latex_escape(str(int(float(row["epsilon"]))))
            rows.append(
                [
                    latex_escape(row["condition_label"]),
                    epsilon,
                    latex_escape(str(int(row["seed_count"]))),
                    plus_minus(row["RMSE_mean"], row["RMSE_std"]),
                    plus_minus(row["MAE_mean"], row["MAE_std"]),
                ]
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-group-dir", default=str(DEFAULT_RUN_GROUP), help="Run-group directory with aggregate CSVs.")
    parser.add_argument(
        "--stem",
        default="table3b_multiseed_privacy_metrics",
        help="Output stem for CSV and TeX summary files.",
    )
    args = parser.parse_args()

    run_group_dir = Path(args.run_group_dir).resolve()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_combined_summary(run_group_dir)
    csv_path = TABLE_DIR / f"{args.stem}.csv"
    tex_path = TABLE_DIR / f"{args.stem}.tex"
    summary.to_csv(csv_path, index=False)

    tex = format_table(
        caption="Five-seed matched-window privacy-utility summary",
        label="tab:multiseed_privacy_metrics",
        colspec=r">{\raggedright\arraybackslash}X>{\centering\arraybackslash}p{0.08\textwidth}>{\centering\arraybackslash}p{0.08\textwidth}>{\raggedleft\arraybackslash}p{0.17\textwidth}>{\raggedleft\arraybackslash}p{0.17\textwidth}",
        header=["Condition / run family", r"$\epsilon$", "Seeds", r"RMSE mean $\pm$ std", r"MAE mean $\pm$ std"],
        rows=build_tex_rows(summary),
        notes=[
            latex_escape(
                f"Built from the aggregated five-seed run group at {run_group_dir.relative_to(ROOT)}."
            ),
            "All rows use the same matched-window contract and the same long-train thesis-facing DP sweep at $w=7$.",
        ],
        size_cmd=r"\footnotesize",
    )
    tex_path.write_text(tex, encoding="utf-8")

    print("Created multiseed privacy summary tables:")
    print(f"- {csv_path.relative_to(ROOT)}")
    print(f"- {tex_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

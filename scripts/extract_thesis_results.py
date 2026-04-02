import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ASOF = "2020-08-05"
OUTPUT_DIR = Path("outputs") / "nyc" / ASOF
TABLE_DIR = OUTPUT_DIR / "thesis_tables"

NONPRIVATE_RUNS = [
    "public_only_adapter_w0_matched_ot",
    "public_only_adapter_w3_matched_ot",
    "public_only_adapter_w7_matched_ot",
    "public_opentable_adapter_w0_matched_ot",
    "public_opentable_adapter_w3_matched_ot",
    "public_opentable_adapter_w7_matched_ot",
]

EVENT_RUNS = [
    "public_opentable_dp_gaussian_event_eps1_w7_matched_ot",
    "public_opentable_dp_gaussian_event_eps2_w7_matched_ot",
    "public_opentable_dp_gaussian_event_eps4_w7_matched_ot",
    "public_opentable_dp_gaussian_event_eps8_w7_matched_ot",
    "public_opentable_dp_gaussian_event_eps16_w7_matched_ot",
]

RESTAURANT_RUNS = [
    "public_opentable_dp_gaussian_restaurant_eps1_w7_matched_ot",
    "public_opentable_dp_gaussian_restaurant_eps2_w7_matched_ot",
    "public_opentable_dp_gaussian_restaurant_eps4_w7_matched_ot",
    "public_opentable_dp_gaussian_restaurant_eps8_w7_matched_ot",
    "public_opentable_dp_gaussian_restaurant_eps16_w7_matched_ot",
]

ALL_RUNS = NONPRIVATE_RUNS + EVENT_RUNS + RESTAURANT_RUNS


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stringify(value):
    if value is None:
        return ""
    return value


def note_join(parts: list[str]) -> str:
    return "; ".join(part for part in parts if part)


def run_paths(run_name: str, smooth_window: int | None) -> dict[str, str]:
    metrics_json = OUTPUT_DIR / f"metrics_{run_name}.json"
    run_metadata_json = OUTPUT_DIR / f"run_metadata_{run_name}.json"
    fit_plot = OUTPUT_DIR / f"fit_train_test_{run_name}.png"
    fit_csv = OUTPUT_DIR / f"fit_train_test_{run_name}.csv"
    forecast_plot = OUTPUT_DIR / f"forecast_vs_truth_{run_name}.png"
    forecast_csv = ""
    if smooth_window is not None:
        candidate = OUTPUT_DIR / f"forecast_28d_w{int(smooth_window)}_{run_name}.csv"
        if candidate.exists():
            forecast_csv = str(candidate.as_posix())

    return {
        "metrics_json_path": str(metrics_json.as_posix()) if metrics_json.exists() else "",
        "run_metadata_json_path": str(run_metadata_json.as_posix()) if run_metadata_json.exists() else "",
        "forecast_plot_path": str(forecast_plot.as_posix()) if forecast_plot.exists() else "",
        "fit_plot_path": str(fit_plot.as_posix()) if fit_plot.exists() else "",
        "forecast_csv_path": forecast_csv,
        "fit_csv_path": str(fit_csv.as_posix()) if fit_csv.exists() else "",
    }


def condition_label_for_run(run_name: str) -> str:
    if run_name.startswith("public_only"):
        return "A: public-only baseline"
    if "_dp_" in run_name:
        return "C: public + OpenTable with Gaussian DP"
    return "B: public + OpenTable (non-private)"


def privacy_mode_label(run_name: str, metrics: dict) -> str:
    value = metrics.get("privacy_mode")
    if value in (None, "", "none"):
        return "none"
    return str(value)


def qualitative_note(run_name: str, rmse: float | None) -> str:
    if rmse is None:
        return "Missing RMSE."
    if run_name == "public_opentable_adapter_w7_matched_ot":
        return "Best non-private RMSE in the matched A/B comparison."
    if run_name == "public_only_adapter_w7_matched_ot":
        return "Best public-only operating point among the matched smoothing runs."
    if run_name == "public_only_adapter_w0_matched_ot":
        return "Better than the public-only w=3 run, but weaker than the public-only w=7 run."
    if run_name == "public_opentable_adapter_w0_matched_ot":
        return "Improves over the public-only w=0 baseline, but is weaker than the OpenTable w=7 run."
    if run_name.endswith("_w3_matched_ot"):
        return "Very unstable relative to w=0 and w=7."
    return ""


def format_markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep] + body)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    missing_fields: dict[str, list[str]] = {}
    warnings: list[str] = []

    extracted: dict[str, dict] = {}
    for run_name in ALL_RUNS:
        metrics_path = OUTPUT_DIR / f"metrics_{run_name}.json"
        metadata_path = OUTPUT_DIR / f"run_metadata_{run_name}.json"
        metrics = read_json(metrics_path)
        if metrics is None:
            raise FileNotFoundError(f"Required metrics JSON missing: {metrics_path}")
        metadata = read_json(metadata_path)
        extracted[run_name] = {"metrics": metrics, "metadata": metadata}

        run_missing: list[str] = []
        if metadata is None:
            run_missing.append("run_metadata_json")
        for key in ("window_start", "window_end", "train_start", "train_end", "test_start", "test_end"):
            if metrics.get(key) in (None, ""):
                run_missing.append(key)
        for key in ("mse", "rmse", "mae", "mape"):
            if metrics.get("test_metrics", {}).get(key) in (None, ""):
                run_missing.append(f"test_metrics.{key}")
        if run_missing:
            missing_fields[run_name] = sorted(set(run_missing))

    for run_name in ALL_RUNS:
        if extracted[run_name]["metadata"] is None:
            warnings.append(f"Missing run metadata JSON for {run_name}; metrics JSON used for table fields.")

    nonprivate_rows: list[dict] = []
    for run_name in NONPRIVATE_RUNS:
        metrics = extracted[run_name]["metrics"]
        test_metrics = metrics.get("test_metrics", {})
        notes = []
        if extracted[run_name]["metadata"] is None:
            notes.append("Run metadata JSON missing.")
        if run_name.startswith("public_only"):
            notes.append("Public-only matched baseline.")
        else:
            notes.append("Non-private OpenTable matched run.")
        nonprivate_rows.append(
            {
                "run_name": run_name,
                "condition_label": condition_label_for_run(run_name),
                "smoothing_window": stringify(metrics.get("smooth_cases_window")),
                "window_start": stringify(metrics.get("window_start")),
                "window_end": stringify(metrics.get("window_end")),
                "train_start": stringify(metrics.get("train_start")),
                "train_end": stringify(metrics.get("train_end")),
                "test_start": stringify(metrics.get("test_start")),
                "test_end": stringify(metrics.get("test_end")),
                "mse": stringify(test_metrics.get("mse")),
                "rmse": stringify(test_metrics.get("rmse")),
                "mae": stringify(test_metrics.get("mae")),
                "mape": stringify(test_metrics.get("mape")),
                "notes": note_join(notes),
            }
        )

    dp_rows: list[dict] = []
    for run_name in EVENT_RUNS + RESTAURANT_RUNS:
        metrics = extracted[run_name]["metrics"]
        metadata = extracted[run_name]["metadata"] or {}
        test_metrics = metrics.get("test_metrics", {})
        notes = []
        if extracted[run_name]["metadata"] is None:
            notes.append("Run metadata JSON missing.")
        private_meta = metadata.get("private_metadata") if isinstance(metadata.get("private_metadata"), dict) else {}
        if private_meta.get("noise_added") is False:
            notes.append("DP metadata inconsistent: noise_added=false.")
        dp_rows.append(
            {
                "run_name": run_name,
                "privacy_mode": stringify(metrics.get("privacy_mode")),
                "epsilon": stringify(metrics.get("epsilon")),
                "delta": stringify(metrics.get("delta")),
                "K": stringify(metrics.get("K")),
                "clipping_bound_pp": stringify(metrics.get("clipping_bound_pp")),
                "sensitivity_day_pp": stringify(metrics.get("sensitivity_day_pp")),
                "sensitivity_l2_pp": stringify(metrics.get("sensitivity_l2_pp")),
                "sigma_pp": stringify(metrics.get("sigma_pp")),
                "window_start": stringify(metrics.get("window_start")),
                "window_end": stringify(metrics.get("window_end")),
                "train_start": stringify(metrics.get("train_start")),
                "train_end": stringify(metrics.get("train_end")),
                "test_start": stringify(metrics.get("test_start")),
                "test_end": stringify(metrics.get("test_end")),
                "mse": stringify(test_metrics.get("mse")),
                "rmse": stringify(test_metrics.get("rmse")),
                "mae": stringify(test_metrics.get("mae")),
                "mape": stringify(test_metrics.get("mape")),
                "notes": note_join(notes),
            }
        )

    main_runs = [
        "public_only_adapter_w7_matched_ot",
        "public_opentable_adapter_w7_matched_ot",
        *EVENT_RUNS,
        *RESTAURANT_RUNS,
    ]
    main_rows: list[dict] = []
    for run_name in main_runs:
        metrics = extracted[run_name]["metrics"]
        test_metrics = metrics.get("test_metrics", {})
        notes = []
        if run_name == "public_only_adapter_w7_matched_ot":
            notes.append("Public-only matched baseline.")
        elif run_name == "public_opentable_adapter_w7_matched_ot":
            notes.append("Non-private OpenTable reference.")
        elif "event" in run_name:
            notes.append("Event-level Gaussian DP.")
        elif "restaurant" in run_name:
            notes.append("Restaurant-level Gaussian DP.")
        main_rows.append(
            {
                "condition_label": condition_label_for_run(run_name),
                "privacy_mode": privacy_mode_label(run_name, metrics),
                "epsilon": stringify(metrics.get("epsilon")),
                "rmse": stringify(test_metrics.get("rmse")),
                "mae": stringify(test_metrics.get("mae")),
                "mape": stringify(test_metrics.get("mape")),
                "sigma_pp": stringify(metrics.get("sigma_pp")),
                "K": stringify(metrics.get("K")),
                "notes": note_join(notes),
            }
        )

    smoothing_rows: list[dict] = []
    for run_name in NONPRIVATE_RUNS:
        metrics = extracted[run_name]["metrics"]
        test_metrics = metrics.get("test_metrics", {})
        smoothing_rows.append(
            {
                "condition_label": condition_label_for_run(run_name),
                "smoothing_window": stringify(metrics.get("smooth_cases_window")),
                "rmse": stringify(test_metrics.get("rmse")),
                "mae": stringify(test_metrics.get("mae")),
                "mape": stringify(test_metrics.get("mape")),
                "qualitative_note": qualitative_note(run_name, test_metrics.get("rmse")),
            }
        )

    manifest_rows: list[dict] = []
    for run_name in ALL_RUNS:
        metrics = extracted[run_name]["metrics"]
        row = {"run_name": run_name}
        row.update(run_paths(run_name, metrics.get("smooth_cases_window")))
        manifest_rows.append(row)

    def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: stringify(row.get(key)) for key in fieldnames})

    write_csv(
        TABLE_DIR / "nonprivate_results_table.csv",
        [
            "run_name",
            "condition_label",
            "smoothing_window",
            "window_start",
            "window_end",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "mse",
            "rmse",
            "mae",
            "mape",
            "notes",
        ],
        nonprivate_rows,
    )
    write_csv(
        TABLE_DIR / "dp_results_table.csv",
        [
            "run_name",
            "privacy_mode",
            "epsilon",
            "delta",
            "K",
            "clipping_bound_pp",
            "sensitivity_day_pp",
            "sensitivity_l2_pp",
            "sigma_pp",
            "window_start",
            "window_end",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "mse",
            "rmse",
            "mae",
            "mape",
            "notes",
        ],
        dp_rows,
    )
    write_csv(
        TABLE_DIR / "final_metrics_main_table.csv",
        [
            "condition_label",
            "privacy_mode",
            "epsilon",
            "rmse",
            "mae",
            "mape",
            "sigma_pp",
            "K",
            "notes",
        ],
        main_rows,
    )
    write_csv(
        TABLE_DIR / "smoothing_comparison_table.csv",
        [
            "condition_label",
            "smoothing_window",
            "rmse",
            "mae",
            "mape",
            "qualitative_note",
        ],
        smoothing_rows,
    )
    write_csv(
        TABLE_DIR / "artifact_manifest.csv",
        [
            "run_name",
            "metrics_json_path",
            "run_metadata_json_path",
            "forecast_plot_path",
            "fit_plot_path",
            "forecast_csv_path",
            "fit_csv_path",
        ],
        manifest_rows,
    )

    nonprivate_best = min(
        nonprivate_rows,
        key=lambda row: float(row["rmse"]) if row["rmse"] != "" else float("inf"),
    )
    event_rows = [row for row in dp_rows if row["privacy_mode"] == "event"]
    restaurant_rows = [row for row in dp_rows if row["privacy_mode"] == "restaurant"]
    best_event = min(event_rows, key=lambda row: float(row["rmse"]))
    best_restaurant = min(restaurant_rows, key=lambda row: float(row["rmse"]))

    preview_lines = [
        "# Thesis Results Preview",
        "",
        "Canonical matched experiment:",
        f"- Window: 2020-02-29 to 2020-08-05",
        f"- Train: 2020-02-29 to 2020-07-08",
        f"- Test: 2020-07-09 to 2020-08-05",
        f"- Forecast horizon: 28 days",
        f"- K: 159",
        "",
        "## Main Thesis Table",
        "",
        format_markdown_table(
            main_rows,
            ["condition_label", "privacy_mode", "epsilon", "rmse", "mae", "mape", "sigma_pp", "K", "notes"],
        ),
        "",
        "## Non-Private Matched Runs",
        "",
        format_markdown_table(
            nonprivate_rows,
            ["run_name", "condition_label", "smoothing_window", "rmse", "mae", "mape", "notes"],
        ),
        "",
        "## DP Matched Runs",
        "",
        format_markdown_table(
            dp_rows,
            ["run_name", "privacy_mode", "epsilon", "rmse", "mae", "mape", "sigma_pp", "notes"],
        ),
        "",
        "## Extraction Notes",
        "",
        f"- Best non-private RMSE in the matched comparison: `{nonprivate_best['run_name']}` with RMSE `{nonprivate_best['rmse']}`.",
        "- `w=7` is the most thesis-facing operating point because it delivers the strongest matched non-private result and avoids the severe instability seen at `w=3`; `w=0` is usable but weaker than the `w=7` OpenTable run.",
        f"- Event-level DP is mixed rather than monotone: the best event-level result is `{best_event['run_name']}` with RMSE `{best_event['rmse']}`, while other event epsilons are weaker than the non-private reference.",
        f"- Restaurant-level DP is also mixed: it beats the non-private OpenTable RMSE at eps `1`, `2`, and `16`, but underperforms event-level DP at eps `4` and `8`.",
        "- Directionally, the event-level and restaurant-level DP curves should be described as non-monotonic and implementation-specific rather than as a simple smooth privacy-utility frontier.",
    ]
    (TABLE_DIR / "thesis_results_preview.md").write_text("\n".join(preview_lines) + "\n", encoding="utf-8")

    caption_lines = [
        "# Figure Caption Numbers",
        "",
        "## Main w=7 non-private comparison",
        "",
        f"- `public_only_adapter_w7_matched_ot`: RMSE `{extracted['public_only_adapter_w7_matched_ot']['metrics']['test_metrics'].get('rmse')}`, MAE `{extracted['public_only_adapter_w7_matched_ot']['metrics']['test_metrics'].get('mae')}`.",
        f"- `public_opentable_adapter_w7_matched_ot`: RMSE `{extracted['public_opentable_adapter_w7_matched_ot']['metrics']['test_metrics'].get('rmse')}`, MAE `{extracted['public_opentable_adapter_w7_matched_ot']['metrics']['test_metrics'].get('mae')}`.",
        "",
        "## Event-level DP RMSE by epsilon",
        "",
        *[
            f"- eps `{row['epsilon']}`: RMSE `{row['rmse']}`."
            for row in event_rows
        ],
        "",
        "## Restaurant-level DP RMSE by epsilon",
        "",
        *[
            f"- eps `{row['epsilon']}`: RMSE `{row['rmse']}`."
            for row in restaurant_rows
        ],
        "",
        "## Factual summaries",
        "",
        "- A/B smoothing comparison: In the matched non-private runs, `w=7` is the strongest operating point for the OpenTable condition, while `w=3` is markedly unstable for both public-only and OpenTable runs.",
        "- Event-level DP trend: Event-level RMSE is non-monotonic across epsilon, with the strongest committed event-level result at eps `4` and weaker results at eps `1`, `2`, `8`, and `16`.",
        "- Restaurant-level DP trend: Restaurant-level RMSE is also non-monotonic, with relatively strong committed results at eps `1`, `2`, and `16`, and weaker results at eps `4` and `8`.",
    ]
    (TABLE_DIR / "figure_caption_numbers.md").write_text("\n".join(caption_lines) + "\n", encoding="utf-8")

    metadata = {
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "runs_included": ALL_RUNS,
        "missing_fields": missing_fields,
        "source_of_truth": "per-run metrics json + run metadata json",
        "warnings": warnings,
    }
    with open(TABLE_DIR / "results_extraction_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote thesis tables to: {TABLE_DIR}")


if __name__ == "__main__":
    main()

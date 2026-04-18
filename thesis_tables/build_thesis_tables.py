import csv
import json
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "thesis_tables"
ASOF = "2020-08-05"
OUTPUT_DIR = ROOT / "outputs" / "nyc" / ASOF

NONPRIVATE_RUNS = [
    "public_only_adapter_w0_matched_ot",
    "public_opentable_adapter_w0_matched_ot",
    "public_only_adapter_w3_matched_ot",
    "public_opentable_adapter_w3_matched_ot",
    "public_only_adapter_w7_matched_ot",
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_str(value) -> str:
    return f"{float(value):.4f}"


def read_csv_shape(path: Path) -> tuple[int, int]:
    df = pd.read_csv(path)
    return int(df.shape[0]), int(df.shape[1])


def load_feature_map(path: Path) -> list[tuple[str, str]]:
    df = pd.read_csv(path)
    return [(str(row["pub_col"]), str(row["source_col"])) for _, row in df.iterrows()]


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


def table1() -> tuple[str, list[str], list[str]]:
    master = ROOT / "data/processed/nyc_master_daily.csv"
    opentable = ROOT / "data/processed/opentable_yoy_daily.csv"
    pop = ROOT / "data/processed/population_nyc_age16_2020.csv"
    contact = ROOT / "data/processed/contact_matrix_us.csv"
    fmap = ROOT / "data/processed/online/public_feature_map_2020-08-05_matched_ot_w7.csv"
    train = ROOT / "data/processed/online/train_2020-08-05_matched_ot_w7.csv"
    private_series = ROOT / "data/processed/private/opentable_private_observed_2020-08-05_matched_ot_series.csv"
    private_meta = ROOT / "data/processed/private/opentable_private_observed_2020-08-05_matched_ot.json"
    private_pt = ROOT / "data/processed/private/opentable_private_observed_2020-08-05_matched_ot.pt"

    master_shape = read_csv_shape(master)
    opentable_shape = read_csv_shape(opentable)
    pop_shape = read_csv_shape(pop)
    contact_df = pd.read_csv(contact)
    contact_shape = (contact_df.shape[0], contact_df.select_dtypes(include="number").shape[1])
    fmap_rows = load_feature_map(fmap)
    train_shape = read_csv_shape(train)
    private_series_shape = read_csv_shape(private_series)
    private_meta_json = read_json(private_meta)
    private_tensor_shape = tuple(private_meta_json["tensor_shape"])
    private_tensor = torch.load(private_pt, map_location="cpu")
    assert tuple(private_tensor.shape) == private_tensor_shape

    covariate_list = ", ".join(src for _, src in fmap_rows)
    rows = [
        [
            r"Public target / master daily table",
            latex_escape(f"Daily panel ({master_shape[0]} rows, {master_shape[1]} columns)"),
            r"Integrates cases, hospitalizations, deaths, mobility, and trend signals in one contiguous public daily table.",
            r"Supplies the case target and the source columns for public features.",
        ],
        [
            r"Public covariates",
            latex_escape(f"Daily matched design; 10 mapped covariates over {train_shape[0]} train days"),
            latex_escape(
                "Numeric public predictors are selected from the master table, excluding date and target fields, then mapped to pub_0..pub_9."
            ),
            latex_escape(f"Public encoder input; feature map resolves the covariate set: {covariate_list}."),
        ],
        [
            r"OpenTable city-level YoY signal",
            latex_escape(f"Daily city series ({opentable_shape[0]} rows)"),
            r"Reduced to the observed \texttt{yoy\_seated\_diner} series before matched-window restriction.",
            r"City-level private-source signal before clipping, scaling, and patch lifting.",
        ],
        [
            r"Patch-aligned OpenTable tensor / private branch representation",
            latex_escape(f"Matched daily release ({private_series_shape[0]} days); tensor {private_tensor_shape[0]}x{private_tensor_shape[1]}"),
            r"Restricted to the matched overlap, clipped to \(\pm 100\) percentage points, scaled by the fixed clip bound, then lifted to 16 patches with population-share weights.",
            r"Private-branch input for non-private and DP OpenTable runs.",
        ],
        [
            r"Population vector",
            latex_escape(f"{pop_shape[0]} patch entries"),
            r"Loaded as the 16-patch population vector for seed scaling and private-signal lifting.",
            r"Sets patch populations and lifting weights.",
        ],
        [
            r"Contact matrix",
            latex_escape(f"{contact_shape[0]}x{contact_shape[1]} matrix"),
            r"Row-normalized at load time after reading the US age-contact matrix.",
            r"Metapopulation coupling matrix for the simulator.",
        ],
    ]
    tex = format_table(
        caption="NYC data contract and model-facing artifacts",
        label="tab:data_contract",
        colspec=r">{\raggedright\arraybackslash}p{0.21\textwidth}>{\raggedright\arraybackslash}p{0.18\textwidth}>{\raggedright\arraybackslash}p{0.29\textwidth}>{\raggedright\arraybackslash}X",
        header=["Artifact", "Temporal resolution / shape", "Preprocessing / transformation", "Role in pipeline"],
        rows=rows,
        size_cmd=r"\footnotesize",
    )
    return tex, [str(p.relative_to(ROOT)) for p in [master, train, fmap, opentable, private_series, private_meta, private_pt, pop, contact]], [
        "Public covariate schema is inferred from the committed public feature map and matched train artifact rather than restated in config.",
    ]


def table2() -> tuple[str, list[str], list[str]]:
    split_w0 = read_json(ROOT / "data/processed/online/split_info_2020-08-05_matched_ot_w0.json")
    split_w3 = read_json(ROOT / "data/processed/online/split_info_2020-08-05_matched_ot_w3.json")
    split_w7 = read_json(ROOT / "data/processed/online/split_info_2020-08-05_matched_ot_w7.json")
    rows = [
        [
            r"Public-only baseline",
            r"Public target history + public covariates",
            r"None",
            r"$w \in \{0,3,7\}$",
        ],
        [
            r"Non-private OpenTable",
            r"Public target history + public covariates + OpenTable tensor",
            r"None",
            r"$w \in \{0,3,7\}$",
        ],
        [
            r"Event-level Gaussian DP OpenTable",
            r"Public target history + public covariates + DP OpenTable tensor",
            r"Gaussian DP (event)",
            r"$w=7$, $\epsilon \in \{1,2,4,8,16\}$",
        ],
        [
            r"Restaurant-level Gaussian DP OpenTable",
            r"Public target history + public covariates + DP OpenTable tensor",
            r"Gaussian DP (restaurant)",
            r"$w=7$, $\epsilon \in \{1,2,4,8,16\}$",
        ],
    ]
    tex = format_table(
        caption="Experimental conditions and comparison ladder",
        label="tab:experimental_conditions",
        colspec=r">{\raggedright\arraybackslash}p{0.25\textwidth}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{0.17\textwidth}>{\raggedright\arraybackslash}p{0.18\textwidth}",
        header=["Condition", "Inputs", "Privacy mode", "Setting"],
        rows=rows,
        notes=[
            latex_escape(
                f"All conditions use the same matched-window contract: matched window {split_w7['window_start']} to {split_w7['window_end']}; "
                f"train {split_w7['train_start']} to {split_w7['train_end']}; test {split_w7['test_start']} to {split_w7['test_end']}; "
                f"{split_w7['test_days']}-day forecast horizon."
            )
        ],
        size_cmd=r"\footnotesize",
    )
    return tex, [
        "data/processed/online/split_info_2020-08-05_matched_ot_w0.json",
        "data/processed/online/split_info_2020-08-05_matched_ot_w3.json",
        "data/processed/online/split_info_2020-08-05_matched_ot_w7.json",
        "scripts/run_matched_ab_grid.sh",
        "scripts/run_matched_dp_grid_w7.sh",
        "src/nyc_gradmeta/utils.py",
    ], [
        "Condition names and run-family labels are compacted from the committed shell scripts and run-tag naming helpers.",
    ]


def grouped_metric_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    groups = [
        ("Non-private matched runs", NONPRIVATE_RUNS),
        ("Event-level Gaussian DP (w=7)", EVENT_RUNS),
        ("Restaurant-level Gaussian DP (w=7)", RESTAURANT_RUNS),
    ]
    label_map = {
        "public_only_adapter_w0_matched_ot": "Public only",
        "public_opentable_adapter_w0_matched_ot": "Public + OpenTable",
        "public_only_adapter_w3_matched_ot": "Public only",
        "public_opentable_adapter_w3_matched_ot": "Public + OpenTable",
        "public_only_adapter_w7_matched_ot": "Public only",
        "public_opentable_adapter_w7_matched_ot": "Public + OpenTable",
    }
    for group_name, run_names in groups:
        rows.append([rf"\multicolumn{{6}}{{l}}{{\textit{{{group_name}}}}}"])
        for run_name in run_names:
            metrics = read_json(OUTPUT_DIR / f"metrics_{run_name}.json")
            test = metrics["test_metrics"]
            rows.append(
                [
                    latex_escape(label_map.get(run_name, "Event-DP OpenTable" if "event" in run_name else "Restaurant-DP OpenTable")),
                    latex_escape(str(metrics["smooth_cases_window"])),
                    (latex_escape(str(int(metrics["epsilon"]))) if metrics.get("epsilon") is not None else r"\textemdash"),
                    metric_str(test["rmse"]),
                    metric_str(test["mae"]),
                    metric_str(test["mape"]),
                ]
            )
    return rows


def table3() -> tuple[str, list[str], list[str]]:
    tex = format_table(
        caption="Final matched-window forecasting metrics",
        label="tab:final_metrics",
        colspec=r">{\raggedright\arraybackslash}X>{\centering\arraybackslash}p{0.06\textwidth}>{\centering\arraybackslash}p{0.08\textwidth}>{\raggedleft\arraybackslash}p{0.15\textwidth}>{\raggedleft\arraybackslash}p{0.15\textwidth}>{\raggedleft\arraybackslash}p{0.15\textwidth}",
        header=["Condition / run family", "$w$", r"$\epsilon$", "RMSE", "MAE", "MAPE"],
        rows=grouped_metric_rows(),
        size_cmd=r"\footnotesize",
    )
    return tex, [f"outputs/nyc/2020-08-05/metrics_{run}.json" for run in ALL_RUNS] + ["outputs/nyc/2020-08-05/metrics_summary.csv"], []


def table4() -> tuple[str, list[str], list[str]]:
    rows = [
        [
            r"ParameterNN",
            r"Private tensor, public features, lagged target context",
            r"Weekly epi parameters, seed vector, beta matrix",
            r"Daily histories \(\rightarrow\) weekly parameters",
            r"Maps matched histories into simulator parameters and initial conditions.",
        ],
        [
            r"Metapopulation SEIRM simulator",
            r"Weekly epi parameters, seed vector, beta matrix, population vector, contact matrix",
            r"Daily city-level forecast",
            r"Daily",
            r"Mechanistic forecast backbone.",
        ],
        [
            r"GRU residual adapter",
            r"Base simulator forecast sequence",
            r"Residual correction sequence",
            r"Daily",
            r"Learns an additive correction on top of the mechanistic forecast.",
        ],
        [
            r"Training stage 1 (\texttt{gradmeta})",
            r"ParameterNN + simulator; adapter frozen/off",
            r"Best parameter-network checkpoint",
            r"Full matched train window",
            r"Fits the mechanistic model to the smoothed training target.",
        ],
        [
            r"Training stage 2 (\texttt{adapter})",
            r"Frozen simulator outputs + adapter residual target",
            r"Best adapter checkpoint",
            r"Full matched train window",
            r"Fits the residual adapter on the default smoothed residual target.",
        ],
        [
            r"Training stage 3 (\texttt{together})",
            r"ParameterNN + simulator + adapter jointly",
            r"Final joint checkpoints and evaluation artifacts",
            r"Full matched train window",
            r"Jointly fine-tunes the mechanistic and residual components.",
        ],
    ]
    tex = format_table(
        caption="Forecasting modules and staged training protocol",
        label="tab:module_training",
        colspec=r">{\raggedright\arraybackslash}p{0.20\textwidth}>{\raggedright\arraybackslash}p{0.23\textwidth}>{\raggedright\arraybackslash}p{0.20\textwidth}>{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}X",
        header=["Module / stage", "Main inputs", "Main outputs", "Timescale", "Role"],
        rows=rows,
        size_cmd=r"\footnotesize",
    )
    return tex, [
        "configs/nyc.json",
        "scripts/build_data.sh",
        "src/nyc_gradmeta/models/forecasting_gradmeta_nyc.py",
        "src/nyc_gradmeta/sim/model_utils.py",
    ], [
        "The thesis-facing name `ParameterNN` is mapped to the committed class `CalibNNTwoEncoderThreeOutputs`.",
    ]


def sigma_grid(mode: str) -> str:
    values = []
    for eps in [1, 2, 4, 8, 16]:
        metrics = read_json(OUTPUT_DIR / f"metrics_public_opentable_dp_gaussian_{mode}_eps{eps}_w7_matched_ot.json")
        values.append(rf"$\epsilon={eps}$: {metric_str(metrics['sigma_pp'])}")
    return "; ".join(values)


def table5() -> tuple[str, list[str], list[str]]:
    event = read_json(OUTPUT_DIR / "metrics_public_opentable_dp_gaussian_event_eps1_w7_matched_ot.json")
    restaurant = read_json(OUTPUT_DIR / "metrics_public_opentable_dp_gaussian_restaurant_eps1_w7_matched_ot.json")
    rows = [
        [r"Protected unit", r"One restaurant contribution on one day", r"One restaurant contribution across the full matched window"],
        [r"Release object", r"City-level OpenTable YoY seated-diner series", r"City-level OpenTable YoY seated-diner series"],
        [r"Injection point", r"Gaussian noise added to the clipped city-level series before patch lifting", r"Gaussian noise added to the clipped city-level series before patch lifting"],
        [r"Clip bound", latex_escape(str(int(event["clipping_bound_pp"]))) + r" percentage points", latex_escape(str(int(restaurant["clipping_bound_pp"]))) + r" percentage points"],
        [r"Post-clip", r"Yes; post-noise clip to \(\pm 100\) pp", r"Yes; post-noise clip to \(\pm 100\) pp"],
        [r"Scaling rule", r"Released YoY divided by fixed \texttt{clipping\_bound\_pp}", r"Released YoY divided by fixed \texttt{clipping\_bound\_pp}"],
        [r"$K$", latex_escape(str(event["K"])), latex_escape(str(restaurant["K"]))],
        [r"Epsilon grid", r"$\{1,2,4,8,16\}$", r"$\{1,2,4,8,16\}$"],
        [r"Delta", metric_str(event["delta"]), metric_str(restaurant["delta"])],
        [r"Sensitivity form", r"$\Delta_2 = 0.25$ pp", r"$\Delta_2 = 0.25\sqrt{159}$ pp"],
        [r"Noise calibration", r"$\sigma = \Delta_2\,c(\delta)/\epsilon$", r"$\sigma = \Delta_2\,c(\delta)/\epsilon$"],
    ]
    tex = format_table(
        caption="Differential privacy calibration and implementation",
        label="tab:dp_calibration",
        colspec=r">{\raggedright\arraybackslash}p{0.20\textwidth}>{\raggedright\arraybackslash}p{0.33\textwidth}>{\raggedright\arraybackslash}X",
        header=["Item", "Event-level DP", "Restaurant-level DP"],
        rows=rows,
        notes=[
            latex_escape(
                "Implementation sequence: city-level OpenTable YoY series -> clip -> Gaussian noise -> post-clip -> fixed scaling -> patch lifting."
            ),
            r"$c(\delta)=\sqrt{2\log(1.25/\delta)}$ with $\delta=0.0001$.",
            "Event-level sigma values by epsilon: " + sigma_grid("event"),
            "Restaurant-level sigma values by epsilon: " + sigma_grid("restaurant"),
        ],
        size_cmd=r"\footnotesize",
    )
    return tex, [
        "scripts/build_private_opentable_tensor.py",
        "data/processed/online/split_info_2020-08-05_matched_ot_w7.json",
    ] + [f"outputs/nyc/2020-08-05/metrics_{run}.json" for run in EVENT_RUNS + RESTAURANT_RUNS], []


def pattern_status(pattern: str) -> tuple[str, list[Path]]:
    if "{" not in pattern:
        path = ROOT / pattern
        return ("Primary record" if path.exists() else "Missing", [path] if path.exists() else [])
    if pattern == "data/processed/online/split_info_2020-08-05_matched_ot_w{0,3,7}.json":
        paths = [ROOT / f"data/processed/online/split_info_2020-08-05_matched_ot_w{w}.json" for w in (0, 3, 7)]
        return ("Matched split records", paths)
    if pattern == "data/processed/private/opentable_private_observed_dp_gaussian_{event|restaurant}_eps{1,2,4,8,16}_2020-08-05_matched_ot_series.csv":
        paths = sorted((ROOT / "data/processed/private").glob("opentable_private_observed_dp_gaussian_*_2020-08-05_matched_ot_series.csv"))
        return ("DP release family", paths)
    raise ValueError(f"Unhandled pattern: {pattern}")


def table_a1() -> tuple[str, list[str], list[str]]:
    rows = []
    artifacts = [
        ("configs/nyc.json", "Experiment configuration and path contract", "Tables 2, 4, and 5", "Primary configuration record"),
        ("scripts/build_data.sh", "Public-data build entrypoint", "Table 1 and Appendix A1", "Canonical public-data wrapper"),
        ("data/processed/nyc_master_daily.csv", "Canonical public daily panel", "Tables 1 and 2", "Master public data table"),
        ("data/processed/opentable_yoy_daily.csv", "Processed city-level OpenTable signal", "Tables 1, 2, and 5", "Canonical OpenTable series"),
        ("scripts/prepare_online_nyc.py and data/processed/online/*", "Matched-window public train/test artifacts and feature maps", "Tables 1 and 2", "Primary matched-window preparation record"),
        ("data/processed/online/split_info_2020-08-05_matched_ot_w{0,3,7}.json", "Matched-window evaluation contract", "Table 2", "Matched-window split records"),
        ("scripts/build_private_opentable_tensor.py", "DP and non-private OpenTable tensor builder", "Tables 1 and 5", "Canonical private-signal builder"),
        ("data/processed/private/opentable_private_observed_2020-08-05_matched_ot_series.csv", "Non-private released OpenTable series", "Tables 1 and 5", "Non-private release record"),
        ("data/processed/private/opentable_private_observed_dp_gaussian_{event|restaurant}_eps{1,2,4,8,16}_2020-08-05_matched_ot_series.csv", "DP released series across privacy settings", "Tables 3 and 5", "DP release family"),
        ("scripts/run_matched_ab_grid.sh", "Non-private comparison launcher", "Table 2 and Section 6 reproduction", "Canonical experiment wrapper"),
        ("scripts/run_matched_dp_grid_w7.sh", "Matched DP sweep launcher", "Tables 2, 3, and 5", "Canonical DP experiment wrapper"),
        ("outputs/nyc/2020-08-05/metrics_<run_tag>.json", "Per-run metrics and DP metadata", "Tables 3 and 5", "Authoritative per-run results record"),
        ("outputs/nyc/2020-08-05/fit_train_test_<run_tag>.csv", "Per-run fitted and held-out trajectories", "Appendix reproducibility audit", "Per-run trajectory record"),
        ("outputs/nyc/2020-08-05/metrics_summary.csv", "Aggregate metrics summary", "Cross-check only", "Convenience summary only"),
        ("scripts/plot_matched_dp_summary.py and outputs/nyc/2020-08-05/rmse_vs_epsilon_*.png", "Privacy-utility summary plotting", "Appendix reproducibility audit", "Summary plotting record"),
    ]
    used_sources: list[str] = []
    for pattern, why, use_in_thesis, status_note in artifacts:
        if pattern == "scripts/prepare_online_nyc.py and data/processed/online/*":
            script = ROOT / "scripts/prepare_online_nyc.py"
            online_paths = sorted((ROOT / "data/processed/online").glob("*"))
            status = "Primary preparation script and artifact family"
            used_sources.extend([str(script.relative_to(ROOT))] + [str(p.relative_to(ROOT)) for p in online_paths])
        elif pattern == "outputs/nyc/2020-08-05/metrics_<run_tag>.json":
            paths = sorted((ROOT / "outputs/nyc/2020-08-05").glob("metrics_*.json"))
            status = "Authoritative run-level metrics family"
            used_sources.extend([str(p.relative_to(ROOT)) for p in paths])
        elif pattern == "outputs/nyc/2020-08-05/fit_train_test_<run_tag>.csv":
            paths = sorted((ROOT / "outputs/nyc/2020-08-05").glob("fit_train_test_*.csv"))
            status = "Run-level trajectory family"
            used_sources.extend([str(p.relative_to(ROOT)) for p in paths])
        elif pattern == "scripts/plot_matched_dp_summary.py and outputs/nyc/2020-08-05/rmse_vs_epsilon_*.png":
            script = ROOT / "scripts/plot_matched_dp_summary.py"
            plots = sorted((ROOT / "outputs/nyc/2020-08-05").glob("rmse_vs_epsilon_*.png"))
            status = "Summary plotting script and figure family"
            used_sources.extend([str(script.relative_to(ROOT))] + [str(p.relative_to(ROOT)) for p in plots])
        else:
            status, paths = pattern_status(pattern)
            used_sources.extend([str(p.relative_to(ROOT)) for p in paths if p.exists()])

        note = status if status_note is None else status_note
        rows.append(
            [
                latex_escape(pattern),
                latex_escape(why),
                latex_escape(use_in_thesis),
                latex_escape(note),
            ]
        )

    tex = format_table(
        caption="Reproducibility artifact checklist",
        label="tab:reproducibility_checklist",
        colspec=r">{\raggedright\arraybackslash}p{0.28\textwidth}>{\raggedright\arraybackslash}p{0.24\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}>{\raggedright\arraybackslash}X",
        header=["Artifact / pattern", "Why it matters", "Use in thesis", "Status note"],
        rows=rows,
        size_cmd=r"\footnotesize",
    )
    return tex, sorted(set(used_sources)), [
        "Pattern rows are compacted into presence counts so the appendix table stays thesis-readable.",
    ]


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "table1_data_contract.tex": table1(),
        "table2_experimental_conditions.tex": table2(),
        "table3_final_metrics.tex": table3(),
        "table4_module_training_summary.tex": table4(),
        "table5_dp_calibration_summary.tex": table5(),
        "tableA1_reproducibility_artifact_checklist.tex": table_a1(),
    }

    summary = []
    for filename, (content, sources, inferences) in outputs.items():
        path = TABLE_DIR / filename
        path.write_text(content, encoding="utf-8")
        summary.append(
            {
                "file": str(path.relative_to(ROOT)),
                "sources": sources,
                "inferences": inferences,
            }
        )

    print("Created table files:")
    for item in summary:
        print(f"- {item['file']}")

    print("\nSource artifacts used for each:")
    for item in summary:
        print(f"- {item['file']}")
        for source in item["sources"]:
            print(f"  - {source}")

    print("\nInferences / compacting decisions:")
    for item in summary:
        if item["inferences"]:
            print(f"- {item['file']}")
            for note in item["inferences"]:
                print(f"  - {note}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CANONICAL_POP_BINS = [
    "0-4",
    "5-9",
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75+",
]

CONTACT_BINS = [
    "00-05",
    "05-10",
    "10-15",
    "15-20",
    "20-25",
    "25-30",
    "30-35",
    "35-40",
    "40-45",
    "45-50",
    "50-55",
    "55-60",
    "60-65",
    "65-70",
    "70-75",
    "75+",
]

AGE_ALIASES = {
    "0-4": "0-4",
    "00-05": "0-4",
    "0-5": "0-4",
    "00_05": "0-4",
    "5-9": "5-9",
    "05-10": "5-9",
    "5-10": "5-9",
    "05_10": "5-9",
    "10-14": "10-14",
    "10-15": "10-14",
    "10_15": "10-14",
    "15-19": "15-19",
    "15-20": "15-19",
    "15_20": "15-19",
    "20-24": "20-24",
    "20-25": "20-24",
    "20_25": "20-24",
    "25-29": "25-29",
    "25-30": "25-29",
    "25_30": "25-29",
    "30-34": "30-34",
    "30-35": "30-34",
    "30_35": "30-34",
    "35-39": "35-39",
    "35-40": "35-39",
    "35_40": "35-39",
    "40-44": "40-44",
    "40-45": "40-44",
    "40_45": "40-44",
    "45-49": "45-49",
    "45-50": "45-49",
    "45_50": "45-49",
    "50-54": "50-54",
    "50-55": "50-54",
    "50_55": "50-54",
    "55-59": "55-59",
    "55-60": "55-59",
    "55_60": "55-59",
    "60-64": "60-64",
    "60-65": "60-64",
    "60_65": "60-64",
    "65-69": "65-69",
    "65-70": "65-69",
    "65_70": "65-69",
    "70-74": "70-74",
    "70-75": "70-74",
    "70_75": "70-74",
    "75+": "75+",
    "75PLUS": "75+",
    "75_PLUS": "75+",
}

CANONICAL_TO_CONTACT = dict(zip(CANONICAL_POP_BINS, CONTACT_BINS))


def _normalize_age_label(label: object) -> str:
    key = str(label).strip().replace(" ", "").upper()
    if key not in AGE_ALIASES:
        raise ValueError(f"Unsupported age label: {label!r}")
    return AGE_ALIASES[key]


def _find_column(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find {what} column. Available columns: {df.columns.tolist()}")


def build_population_from_patchflow(population_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(population_csv)
    age_col = _find_column(df, ["age_group", "age", "Age", "age_bin", "AgeGroup"], "age")
    pop_col = _find_column(
        df,
        ["population", "Population", "pop", "POPULATION", "value", "count"],
        "population",
    )
    out = df[[age_col, pop_col]].copy()
    out[age_col] = out[age_col].map(_normalize_age_label)
    out[pop_col] = pd.to_numeric(out[pop_col], errors="raise")
    out = out.groupby(age_col, as_index=False)[pop_col].sum()
    out = out.rename(columns={age_col: "age_group", pop_col: "population"})
    out = out.set_index("age_group").reindex(CANONICAL_POP_BINS).reset_index()
    if out["population"].isna().any():
        missing = out.loc[out["population"].isna(), "age_group"].tolist()
        raise ValueError(f"Population CSV is missing required age bins after normalization: {missing}")
    out["population"] = out["population"].astype(int)
    return out


def build_contact_from_patchflow(contact_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(contact_csv, index_col=0)
    df.index = [_normalize_age_label(idx) for idx in df.index]
    df.columns = [_normalize_age_label(col) for col in df.columns]
    df = df.apply(pd.to_numeric, errors="raise")
    df = df.groupby(level=0).sum()
    df = df.T.groupby(level=0).sum().T
    df = df.reindex(index=CANONICAL_POP_BINS, columns=CANONICAL_POP_BINS)
    if df.isna().any().any():
        raise ValueError("Contact matrix is missing one or more required age bins after normalization.")
    df.index = [CANONICAL_TO_CONTACT[idx] for idx in df.index]
    df.columns = [CANONICAL_TO_CONTACT[col] for col in df.columns]
    return df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Normalize age-stratified Patchflow-style inputs into this repo's "
            "16-bin population/contact CSVs."
        )
    )
    ap.add_argument("--population_csv", required=True, help="Local Patchflow-derived population CSV.")
    ap.add_argument("--contact_csv", default=None, help="Optional local Patchflow-derived contact matrix CSV.")
    ap.add_argument(
        "--population_out",
        default="data/processed/population_nyc_age16_patchflow.csv",
        help="Output path for normalized 16-bin population CSV.",
    )
    ap.add_argument(
        "--contact_out",
        default="data/processed/contact_matrix_patchflow.csv",
        help="Output path for normalized 16x16 contact matrix CSV.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]

    population_out = project_root / args.population_out
    population_out.parent.mkdir(parents=True, exist_ok=True)
    population_df = build_population_from_patchflow(project_root / args.population_csv)
    population_df.to_csv(population_out, index=False)
    print(f"Saved normalized population CSV to {population_out}")
    print(f"Population rows: {len(population_df)}")
    print(f"Population total: {int(population_df['population'].sum())}")

    if args.contact_csv:
        contact_out = project_root / args.contact_out
        contact_out.parent.mkdir(parents=True, exist_ok=True)
        contact_df = build_contact_from_patchflow(project_root / args.contact_csv)
        contact_df.to_csv(contact_out)
        print(f"Saved normalized contact matrix CSV to {contact_out}")
        print(f"Contact shape: {contact_df.shape}")


if __name__ == "__main__":
    main()

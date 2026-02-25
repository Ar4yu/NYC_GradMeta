from __future__ import annotations

import pandas as pd
from pathlib import Path
import numpy as np


# -------------------------------------------------
# Paths (relative to repo root)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CASES_PATH = PROCESSED_DIR / "cases_nyc_daily.csv"
MOBILITY_PATH = PROCESSED_DIR / "mobility_nyc_daily.csv"

# You may have one of these depending on what you named it earlier
TRENDS_CANDIDATES = [
    PROCESSED_DIR / "trends_us_ny_501_daily.csv",
    PROCESSED_DIR / "trends_us_ny_daily.csv",
    PROCESSED_DIR / "trends_us_ny_daily_501.csv",
    PROCESSED_DIR / "trends_nyc_daily.csv",
]

OUT_PATH = PROCESSED_DIR / "nyc_master_daily.csv"

END_DATE = pd.Timestamp("2022-12-31")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def _pick_trends_path(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find trends file. Looked for:\n" + "\n".join(str(p) for p in candidates)
    )


def _ensure_date_col(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError(f"{name} must have a 'date' column. Found columns: {list(df.columns)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def _parse_numeric_series(series: pd.Series) -> tuple[pd.Series, int]:
    raw = series.copy()
    cleaned = (
        raw.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    parsed = pd.to_numeric(cleaned, errors="coerce")
    bad_mask = raw.notna() & parsed.isna()
    return parsed, int(bad_mask.sum())


def main() -> None:
    _require_file(CASES_PATH)
    _require_file(MOBILITY_PATH)
    trends_path = _pick_trends_path(TRENDS_CANDIDATES)
    print("Using trends file:", trends_path.name)

    # ---------------------------
    # Load
    # ---------------------------
    cases = pd.read_csv(CASES_PATH)
    mob = pd.read_csv(MOBILITY_PATH)
    trends = pd.read_csv(trends_path)

    cases = _ensure_date_col(cases, "cases")
    mob = _ensure_date_col(mob, "mobility")
    trends = _ensure_date_col(trends, "trends")

    # ---------------------------
    # Sanity: expected columns
    # ---------------------------
    # cases expected: date, cases, probable_cases, hospitalizations, deaths
    for c in ["cases", "probable_cases", "hospitalizations", "deaths"]:
        if c not in cases.columns:
            raise ValueError(f"cases file missing column '{c}'. Found: {list(cases.columns)}")

    # mobility expected columns: date + 6 mob_*
    mob_cols = [c for c in mob.columns if c.startswith("mob_")]
    if len(mob_cols) == 0:
        raise ValueError(
            "mobility file has no 'mob_*' columns. Found: "
            f"{list(mob.columns)}"
        )

    # trends: assume 1 signal column besides date
    trend_cols = [c for c in trends.columns if c != "date"]
    if len(trend_cols) != 1:
        raise ValueError(
            f"trends file should have exactly 1 non-date column. Found: {trend_cols}"
        )
    trend_col = trend_cols[0]

    # Standardize trend column name (nice for downstream)
    trends = trends.rename(columns={trend_col: "trend_covid_topic"})

    # ---------------------------
    # Restrict all to <= 2022-12-31
    # ---------------------------
    cases = cases[cases["date"] <= END_DATE].copy()
    mob = mob[mob["date"] <= END_DATE].copy()
    trends = trends[trends["date"] <= END_DATE].copy()

    # ---------------------------
    # Merge (keep only overlapping dates by default)
    # ---------------------------
    df = cases.merge(mob, on="date", how="inner").merge(trends, on="date", how="inner")

    # ---------------------------
    # Enforce continuous daily index on the merged window
    # (start = first merged day, end = 2022-12-31)
    # ---------------------------
    start = df["date"].min()
    end = min(df["date"].max(), END_DATE)

    full = pd.date_range(start, end, freq="D")

    df = (
        df.set_index("date")
          .reindex(full)
          .rename_axis("date")
          .reset_index()
          .sort_values("date")
    )

    # ---------------------------
    # Fill missing values
    # ---------------------------
    # Counts: safest default is 0 if missing (especially after reindex)
    parse_issues = {}
    for c in ["cases", "probable_cases", "hospitalizations", "deaths"]:
        parsed, bad_count = _parse_numeric_series(df[c])
        df[c] = parsed.fillna(0).clip(lower=0)
        parse_issues[c] = bad_count

    # Mobility + trend: interpolate then ffill/bfill (smooth daily covariates)
    cov_cols = mob_cols + ["trend_covid_topic"]
    for c in cov_cols:
        parsed, _ = _parse_numeric_series(df[c])
        df[c] = parsed
        df[c] = df[c].interpolate(limit_direction="both")
        df[c] = df[c].ffill().bfill()

    # Helpful derived column
    df["total_cases"] = df["cases"] + df["probable_cases"]

    # ---------------------------
    # Save
    # ---------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # ---------------------------
    # Report
    # ---------------------------
    print("\nSaved:", OUT_PATH)
    print("Shape:", df.shape)
    print("Min date:", df["date"].min())
    print("Max date:", df["date"].max())
    print("Parse issues in count columns:", parse_issues)
    print("\nColumns:", list(df.columns))
    print("\nHead:")
    print(df.head())
    print("\nTail:")
    print(df.tail())


if __name__ == "__main__":
    main()

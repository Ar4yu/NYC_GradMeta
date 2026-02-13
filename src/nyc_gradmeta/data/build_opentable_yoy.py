"""
Build OpenTable YoY seated diner daily series for NYC (city-level) from the Kaggle
"YoY_Seated_Diner_Data.csv" wide-format file.

Output:
  data/processed/opentable_yoy_daily.csv
  columns: [date, yoy_seated_diner]

Notes:
- Prefers Type="city", Name="New York" (NYC-level) if available.
- Falls back to Type="state", Name="New York" if the city row is missing.
- The dataset is usually wide columns like "2/18", "2/19", ... and often corresponds to 2020.
  If the source ever changes to include years, this script handles that too.
"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "YoY_Seated_Diner_Data.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "opentable_yoy_daily.csv"


# ----------------------------
# Helpers
# ----------------------------
_DATE_COL_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\s*$")


def _parse_wide_date_col(col: str, default_year: int) -> pd.Timestamp | None:
    """
    Parse a wide column header that might be:
      - "2/18"  -> assume default_year
      - "2/18/20" or "2/18/2020"
      - "2020-02-18"
    Returns pd.Timestamp or None if not parseable as a date column.
    """
    col = str(col).strip()

    # ISO-ish
    try:
        ts = pd.to_datetime(col, errors="raise")
        # Guard: if it parses but it's something weird, keep only date-ish
        if ts.year >= 1990 and ts.year <= 2100:
            return ts.normalize()
    except Exception:
        pass

    m = _DATE_COL_RE.match(col)
    if not m:
        return None

    month = int(m.group(1))
    day = int(m.group(2))
    year_s = m.group(3)

    if year_s is None:
        year = default_year
    else:
        year_i = int(year_s)
        year = 2000 + year_i if year_i < 100 else year_i

    return pd.Timestamp(year=year, month=month, day=day)


def _infer_default_year(date_cols: list[str]) -> int:
    """
    If the headers have explicit years, we’ll use them.
    Otherwise, OpenTable Kaggle YoY dataset is commonly from early 2020.
    """
    # If any column looks like it includes a year, let parse handle it (default doesn't matter much).
    for c in date_cols:
        if _DATE_COL_RE.match(str(c).strip()) and _DATE_COL_RE.match(str(c).strip()).group(3):
            return 2020
    # Common case: no year in headers -> assume 2020
    return 2020


def _select_nyc_row(df: pd.DataFrame) -> pd.Series:
    """
    Prefer NYC city row; fallback to NY state row.
    """
    if "Type" not in df.columns or "Name" not in df.columns:
        raise ValueError("Expected columns 'Type' and 'Name' in OpenTable file.")

    # Normalize
    d = df.copy()
    d["Type"] = d["Type"].astype(str).str.strip().str.lower()
    d["Name"] = d["Name"].astype(str).str.strip()

    # Prefer city New York
    city = d[(d["Type"] == "city") & (d["Name"] == "New York")]
    if len(city) >= 1:
        return city.iloc[0]

    # Fallback: state New York
    state = d[(d["Type"] == "state") & (d["Name"] == "New York")]
    if len(state) >= 1:
        return state.iloc[0]

    # If nothing, print some helpful context
    sample_types = sorted(d["Type"].dropna().unique().tolist())[:10]
    ny_like = d[d["Name"].str.contains("New York", na=False)][["Type", "Name"]].drop_duplicates()
    raise RuntimeError(
        "Could not find OpenTable row for NYC.\n"
        f"Types seen (sample): {sample_types}\n"
        f"Rows where Name contains 'New York':\n{ny_like.to_string(index=False)}"
    )


def build_opentable_yoy_daily(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw OpenTable file: {raw_path}")

    df = pd.read_csv(raw_path)

    # Identify date-like wide columns (everything except Type/Name typically)
    candidate_cols = [c for c in df.columns if c not in ("Type", "Name")]
    default_year = _infer_default_year(candidate_cols)

    # Build a mapping col -> Timestamp for parseable date columns
    date_map = {}
    for c in candidate_cols:
        ts = _parse_wide_date_col(c, default_year=default_year)
        if ts is not None:
            date_map[c] = ts

    if not date_map:
        raise RuntimeError(
            "No date columns recognized in OpenTable file. "
            "Expected headers like '2/18' or '2020-02-18'."
        )

    ny_row = _select_nyc_row(df)

    # Pull values for date columns into long form
    records = []
    for col, ts in date_map.items():
        val = ny_row.get(col, None)
        records.append({"date": ts, "yoy_seated_diner": val})

    out = pd.DataFrame(records)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["yoy_seated_diner"] = pd.to_numeric(out["yoy_seated_diner"], errors="coerce")

    out = (
        out.dropna(subset=["date"])
           .groupby("date", as_index=False)["yoy_seated_diner"].mean()
           .sort_values("date")
    )

    # Create a fully daily index (in case the file has gaps)
    full = pd.date_range(out["date"].min(), out["date"].max(), freq="D")
    out = out.set_index("date").reindex(full).rename_axis("date").reset_index()

    # For OpenTable series, interpolation is usually reasonable for small gaps.
    out["yoy_seated_diner"] = out["yoy_seated_diner"].interpolate(limit_direction="both")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    return out


def main() -> None:
    out = build_opentable_yoy_daily()
    print(f"\nSaved: {OUT_PATH}")
    print("Shape:", out.shape)
    print("Min date:", out["date"].min())
    print("Max date:", out["date"].max())
    print(out.head())
    print(out.tail())


if __name__ == "__main__":
    main()

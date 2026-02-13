"""
inspect_data.py

Location: src/nyc_gradmeta/data/inspect_data.py

Two modes:
  1) --mode nyc     -> build daily NYC mobility features from Google's Region Mobility Reports zip
  2) --mode inspect -> lightweight inspection of files inside the zip (optional; can be heavy)

Usage:
  python src/nyc_gradmeta/data/inspect_data.py --mode nyc
  python src/nyc_gradmeta/data/inspect_data.py --mode inspect --max-files 20
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List

import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# This assumes: repo_root/
#   data/raw/Region_Mobility_Report_CSVs.zip
#   src/nyc_gradmeta/data/inspect_data.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ZIP_PATH = DATA_RAW_DIR / "Region_Mobility_Report_CSVs.zip"
ARCHIVE_PATH = ZIP_PATH  # alias (same file)

# -----------------------------------------------------------------------------
# Mobility extraction config
# -----------------------------------------------------------------------------
MOBILITY_COLS: List[str] = [
    "country_region_code",
    "country_region",
    "sub_region_1",
    "sub_region_2",
    "date",
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline",
]

NYC_COUNTIES = {
    "New York County",   # Manhattan
    "Kings County",      # Brooklyn
    "Queens County",
    "Bronx County",
    "Richmond County",   # Staten Island
}

MOBILITY_FEATURES: List[str] = [
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline",
]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def list_zip_members(zip_path: Path) -> List[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.namelist()


def load_us_mobility_year(
    zip_path: Path,
    year: int,
    chunksize: int = 250_000,
) -> Iterator[pd.DataFrame]:
    """
    Stream-read the US mobility CSV for a given year from the zip.
    This avoids loading the entire US file into memory.
    """
    target = f"{year}_US_Region_Mobility_Report.csv"
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        if target not in names:
            raise FileNotFoundError(f"{target} not found inside {zip_path.name}")

        with zf.open(target) as f:
            # usecols lambda keeps only what we need (safe even if extra cols appear)
            reader = pd.read_csv(
                f,
                usecols=lambda c: c in set(MOBILITY_COLS),
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in reader:
                yield chunk


def build_nyc_mobility(zip_path: Path, years: List[int]) -> pd.DataFrame:
    """
    Build a daily time-series for NYC from the US mobility files in the zip.
    Filters to NY state + NYC counties, then averages across counties per day.
    """
    pieces: List[pd.DataFrame] = []

    for y in years:
        for chunk in load_us_mobility_year(zip_path, y):
            # normalize strings (robust to whitespace / missing)
            if "sub_region_1" in chunk.columns:
                chunk["sub_region_1"] = chunk["sub_region_1"].astype(str).str.strip()
            if "sub_region_2" in chunk.columns:
                chunk["sub_region_2"] = chunk["sub_region_2"].astype(str).str.strip()

            nyc_mask = chunk["sub_region_2"].isin(NYC_COUNTIES) | (chunk["sub_region_2"] == "New York City")
            chunk = chunk[(chunk["sub_region_1"] == "New York") & nyc_mask].copy()

            if chunk.empty:
                continue

            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            chunk = chunk.dropna(subset=["date"])

            # average across counties per day
            daily = chunk.groupby("date", as_index=False)[MOBILITY_FEATURES].mean(numeric_only=True)
            pieces.append(daily)

    if not pieces:
        raise RuntimeError(
            "No NYC mobility rows found. Check that the US year files exist in the zip and county names match."
        )

    df = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Rename to shorter names (optional but convenient)
    df = df.rename(
        columns={
            "retail_and_recreation_percent_change_from_baseline": "mob_retail",
            "grocery_and_pharmacy_percent_change_from_baseline": "mob_grocery",
            "parks_percent_change_from_baseline": "mob_parks",
            "transit_stations_percent_change_from_baseline": "mob_transit",
            "workplaces_percent_change_from_baseline": "mob_work",
            "residential_percent_change_from_baseline": "mob_residential",
        }
    )

    return df


# -----------------------------------------------------------------------------
# ZIP inspection (optional / exploratory)
# -----------------------------------------------------------------------------
def _is_text_table(name: str) -> bool:
    name_l = name.lower()
    return any(name_l.endswith(ext) for ext in [".csv", ".tsv", ".txt", ".json", ".jsonl"])


def _is_binary_table(name: str) -> bool:
    name_l = name.lower()
    return any(name_l.endswith(ext) for ext in [".xlsx", ".xls", ".parquet", ".feather"])


def _read_csv_like(file_bytes: bytes, name: str) -> pd.DataFrame:
    # Try common separators and engines; keep it robust.
    buf = io.BytesIO(file_bytes)

    try:
        return pd.read_csv(buf, low_memory=False)
    except Exception:
        pass

    buf = io.BytesIO(file_bytes)
    try:
        return pd.read_csv(buf, sep="\t", low_memory=False)
    except Exception:
        pass

    buf = io.BytesIO(file_bytes)
    return pd.read_csv(buf, sep=None, engine="python", low_memory=False)


def _read_json_like(file_bytes: bytes, name: str) -> pd.DataFrame:
    name_l = name.lower()
    buf = io.BytesIO(file_bytes)

    if name_l.endswith(".jsonl") or "jsonl" in name_l:
        return pd.read_json(buf, lines=True)

    try:
        return pd.read_json(buf)
    except ValueError:
        buf = io.BytesIO(file_bytes)
        return pd.read_json(buf, lines=True)


def _read_binary_table(file_bytes: bytes, name: str) -> pd.DataFrame:
    name_l = name.lower()
    buf = io.BytesIO(file_bytes)

    if name_l.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    if name_l.endswith(".parquet"):
        return pd.read_parquet(buf)
    if name_l.endswith(".feather"):
        return pd.read_feather(buf)

    raise ValueError(f"Unsupported binary table format for {name}")


def summarize_df(df: pd.DataFrame, label: str, max_preview_rows: int = 5) -> None:
    print("\n" + "=" * 90)
    print(f"[DATAFRAME] {label}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")

    print("\nColumns:")
    print(list(df.columns))

    print("\nDtypes (top 30):")
    print(df.dtypes.head(30))

    print("\nMissingness (top 30):")
    miss = df.isna().mean().sort_values(ascending=False)
    print((miss.head(30) * 100).round(2).astype(str) + "%")

    print(f"\nHead({max_preview_rows}):")
    print(df.head(max_preview_rows))

    # robust describe: compatible with older pandas (no numeric_only kwarg)
    print("\nDescribe (numeric):")
    num = df.select_dtypes(include="number")
    if num.shape[1] == 0:
        print("(no numeric columns)")
    else:
        print(num.describe().T.head(20))

    print("=" * 90)


def list_zip_tree(zf: zipfile.ZipFile, prefix: str = "", max_show: int = 200) -> None:
    names = sorted([n for n in zf.namelist() if not n.endswith("/")])
    print(f"\n{prefix}ZIP contains {len(names)} files (excluding folders). Showing up to {max_show}:")
    for n in names[:max_show]:
        info = zf.getinfo(n)
        print(f" - {n}  ({info.file_size / 1024:.1f} KB)")
    if len(names) > max_show:
        print(f" ... and {len(names) - max_show} more")


def read_tables_from_zip(archive_path: Path, max_files_to_read: int = 50, max_preview_rows: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Returns: dict mapping 'path_in_zip' -> DataFrame
    Prints a file listing and per-table summaries.
    Handles nested zips (one level deep) in-memory.

    NOTE: This can still be heavy if you set max_files_to_read too high.
    """
    archive_path = Path(archive_path)

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    dfs: Dict[str, pd.DataFrame] = {}

    print(f"\nOpening archive: {archive_path}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        list_zip_tree(zf)

        members = [n for n in zf.namelist() if not n.endswith("/")]
        read_count = 0

        for name in members:
            if read_count >= max_files_to_read:
                print(f"\nReached max_files_to_read={max_files_to_read}. Stopping.")
                break

            name_l = name.lower()

            # Nested zip handling
            if name_l.endswith(".zip"):
                print(f"\n--- Found nested zip: {name} ---")
                nested_bytes = zf.read(name)
                with zipfile.ZipFile(io.BytesIO(nested_bytes), "r") as nzf:
                    list_zip_tree(nzf, prefix="  [nested] ")

                    nested_members = [nn for nn in nzf.namelist() if not nn.endswith("/")]
                    for nn in nested_members:
                        if read_count >= max_files_to_read:
                            break

                        if _is_text_table(nn) or _is_binary_table(nn):
                            try:
                                b = nzf.read(nn)
                                if _is_text_table(nn):
                                    if nn.lower().endswith((".json", ".jsonl")):
                                        df = _read_json_like(b, nn)
                                    else:
                                        df = _read_csv_like(b, nn)
                                else:
                                    df = _read_binary_table(b, nn)

                                key = f"{name}::{nn}"
                                dfs[key] = df
                                summarize_df(df, key, max_preview_rows=max_preview_rows)
                                read_count += 1
                            except Exception as e:
                                print(f"[WARN] Failed reading {name}::{nn}: {e}")

                continue

            # Top-level table files
            if _is_text_table(name) or _is_binary_table(name):
                try:
                    b = zf.read(name)
                    if _is_text_table(name):
                        if name_l.endswith((".json", ".jsonl")):
                            df = _read_json_like(b, name)
                        else:
                            df = _read_csv_like(b, name)
                    else:
                        df = _read_binary_table(b, name)

                    dfs[name] = df
                    summarize_df(df, name, max_preview_rows=max_preview_rows)
                    read_count += 1
                except Exception as e:
                    print(f"[WARN] Failed reading {name}: {e}")

    print(f"\nDone. Loaded {len(dfs)} table(s) into memory.")
    return dfs


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
def run() -> None:
    parser = argparse.ArgumentParser(description="Inspect and extract NYC mobility data from the Google Region Mobility zip.")
    parser.add_argument(
        "--mode",
        choices=["nyc", "inspect"],
        default="nyc",
        help="nyc = build daily NYC mobility csv; inspect = read+summarize a limited number of files in the zip",
    )
    parser.add_argument("--max-files", type=int, default=20, help="Only used for --mode inspect.")
    parser.add_argument("--preview-rows", type=int, default=5, help="Only used for --mode inspect.")
    parser.add_argument("--chunksize", type=int, default=250_000, help="Chunk size for streaming US mobility csv (nyc mode).")
    args = parser.parse_args()

    # Ensure paths exist
    if not DATA_RAW_DIR.exists():
        raise FileNotFoundError(f"Expected raw data dir missing: {DATA_RAW_DIR}")
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Expected mobility zip missing: {ZIP_PATH}")

    if args.mode == "inspect":
        dfs = read_tables_from_zip(ARCHIVE_PATH, max_files_to_read=args.max_files, max_preview_rows=args.preview_rows)
        inventory = (
            pd.DataFrame([{"key": k, "rows": v.shape[0], "cols": v.shape[1]} for k, v in dfs.items()])
            .sort_values(["rows", "cols"], ascending=False)
            .reset_index(drop=True)
        )
        print("\nInventory (largest first):")
        print(inventory.head(30))

    elif args.mode == "nyc":
        # patch chunksize into loader via a small wrapper
        global load_us_mobility_year  # (kept simple; avoids refactor)
        _orig_loader = load_us_mobility_year

        def _loader(zip_path: Path, year: int, chunksize: int = args.chunksize) -> Iterator[pd.DataFrame]:
            return _orig_loader(zip_path, year, chunksize=chunksize)

        load_us_mobility_year = _loader  # type: ignore

        print("ZIP:", ZIP_PATH)
        members = set(list_zip_members(ZIP_PATH))
        years = [y for y in range(2020, 2027) if f"{y}_US_Region_Mobility_Report.csv" in members]
        print("US years found:", years)

        nyc_mob = build_nyc_mobility(ZIP_PATH, years)
        print("NYC mobility shape:", nyc_mob.shape)
        print(nyc_mob.head())

        out = DATA_PROCESSED_DIR / "mobility_nyc_daily.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        nyc_mob.to_csv(out, index=False)
        print("Wrote:", out)


if __name__ == "__main__":
    run()

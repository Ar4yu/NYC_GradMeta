import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def compute_mobility_index(df: pd.DataFrame) -> pd.Series:
    mob_cols = [c for c in df.columns if c.startswith("mob_")]
    if len(mob_cols) == 0:
        raise ValueError("No mobility columns found (expected columns starting with 'mob_').")
    # simple, stable 1D compression: mean across mobility signals
    return df[mob_cols].mean(axis=1)


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return json.load(f)


def parse_numeric_column(series: pd.Series, col_name: str) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    parsed = pd.to_numeric(cleaned, errors="coerce")
    bad_count = int(series.notna().sum() - parsed.notna().sum())
    if bad_count > 0:
        print(f"[warn] Column '{col_name}' had {bad_count} non-numeric values; filling with 0.")
    return parsed.fillna(0.0)


def build_public_features(
    df: pd.DataFrame,
    mode: str,
    target_col: str,
    target_source_col: str,
    exclude_public_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    exclude = set(exclude_public_columns or [])
    exclude.update({target_col, target_source_col})

    if mode == "mobility_index":
        feat = pd.DataFrame({"pub_0": compute_mobility_index(df).astype(float)})
        mapping = [("pub_0", "mobility_index")]
        return feat, mapping

    if mode == "trend":
        if "trend_covid_topic" not in df.columns:
            raise ValueError("Expected trend column 'trend_covid_topic' not found.")
        feat = pd.DataFrame({"pub_0": df["trend_covid_topic"].astype(float)})
        mapping = [("pub_0", "trend_covid_topic")]
        return feat, mapping

    if mode == "all_public":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        source_cols = [c for c in numeric_cols if c not in exclude]
        if len(source_cols) == 0:
            raise ValueError(
                "No numeric public feature columns left after excluding target/source. "
                f"Numeric columns={numeric_cols}, excluded={sorted(exclude)}"
            )
        feat = df[source_cols].astype(float).copy()
        mapping = []
        renamed_cols = []
        for i, src_col in enumerate(source_cols):
            pub_col = f"pub_{i}"
            renamed_cols.append(pub_col)
            mapping.append((pub_col, src_col))
        feat.columns = renamed_cols
        return feat, mapping

    raise ValueError(
        f"Unknown public_feature_mode='{mode}'. Use 'mobility_index', 'trend', or 'all_public'."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nyc.json")
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD. Test window ends on this date (inclusive).")
    ap.add_argument(
        "--split_mode",
        choices=["horizon", "ratio"],
        default="horizon",
        help="horizon: fixed trailing test window; ratio: chronological train/test ratio split.",
    )
    ap.add_argument(
        "--test_days",
        type=int,
        default=None,
        help="Only used when split_mode=horizon. Defaults to nyc.test_days in config.",
    )
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Only used when split_mode=ratio (chronological split).",
    )
    ap.add_argument(
        "--window_days",
        type=int,
        default=None,
        help="Optional: use only the last N days up to ASOF before splitting (e.g., 365).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    nyc = cfg["nyc"]

    master_path = Path(nyc["paths"]["master_daily_csv"])
    online_dir = Path(nyc["paths"]["online_dir"])
    online_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(master_path)
    if "date" not in df.columns:
        raise ValueError("master_daily_csv must include a 'date' column.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ValueError("master_daily_csv has duplicate dates.")

    diffs = df["date"].diff().dropna().dt.days
    if not (diffs == 1).all():
        raise ValueError("master_daily_csv must be daily contiguous with no missing dates.")

    # Build target column "cases" from configured source column
    src = nyc.get("target_source_column", "total_cases")
    if src not in df.columns:
        raise ValueError(f"Configured target_source_column '{src}' not found in master_daily_csv columns.")
    df["cases"] = parse_numeric_column(df[src], src)

    mode = nyc.get("public_feature_mode", "all_public")
    feature_df, mapping = build_public_features(
        df=df,
        mode=mode,
        target_col="cases",
        target_source_col=src,
        exclude_public_columns=nyc.get("exclude_public_columns", []),
    )
    out = pd.concat([df[["cases"]].astype(float).reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    if out.isna().any().any():
        raise ValueError("Prepared online dataframe contains NaNs.")

    # Restrict to <= ASOF before splitting.
    asof = pd.to_datetime(args.asof)

    # Restrict to <= asof
    mask_upto = df["date"] <= asof
    df_upto = df.loc[mask_upto].copy()
    out_upto = out.loc[mask_upto].reset_index(drop=True)
    if df_upto.empty:
        raise ValueError(f"No rows found in master_daily_csv up to asof={args.asof}.")

    # Find index split based on date range
    dates_upto = df_upto["date"].reset_index(drop=True)
    diffs_upto = dates_upto.diff().dropna().dt.days
    if not (diffs_upto == 1).all():
        raise ValueError("Rows up to asof are not daily contiguous.")

    if args.window_days is not None:
        if args.window_days < 2:
            raise ValueError("--window_days must be >= 2.")
        start_idx = max(0, len(out_upto) - int(args.window_days))
        out_upto = out_upto.iloc[start_idx:].reset_index(drop=True)
        dates_upto = dates_upto.iloc[start_idx:].reset_index(drop=True)
    if len(out_upto) < 2:
        raise ValueError("Need at least 2 rows after ASOF/window filtering to split train/test.")

    if args.split_mode == "horizon":
        test_days = int(args.test_days) if args.test_days is not None else int(nyc["test_days"])
        test_start = asof - pd.Timedelta(days=test_days - 1)
        test_mask = (dates_upto >= test_start) & (dates_upto <= asof)
        if test_mask.sum() != test_days:
            raise ValueError(
                f"Expected exactly {test_days} test days between {test_start.date()} and {asof.date()}, "
                f"found {int(test_mask.sum())}. Check missing dates in master_daily_csv."
            )
        train_df = out_upto.loc[~test_mask].reset_index(drop=True)
        test_df = out_upto.loc[test_mask].reset_index(drop=True)
        split_info = {
            "split_mode": "horizon",
            "test_days": test_days,
            "train_days": int(len(train_df)),
            "window_days": args.window_days,
            "asof": args.asof,
        }
    else:
        ratio = float(args.train_ratio)
        if not (0.0 < ratio < 1.0):
            raise ValueError("--train_ratio must be between 0 and 1.")
        n_total = len(out_upto)
        train_n = int(np.floor(n_total * ratio))
        train_n = max(1, min(train_n, n_total - 1))
        train_df = out_upto.iloc[:train_n].reset_index(drop=True)
        test_df = out_upto.iloc[train_n:].reset_index(drop=True)
        split_info = {
            "split_mode": "ratio",
            "train_ratio": ratio,
            "train_days": int(len(train_df)),
            "test_days": int(len(test_df)),
            "window_days": args.window_days,
            "asof": args.asof,
        }

    # Save
    train_path = online_dir / f"train_{args.asof}.csv"
    test_path = online_dir / f"test_{args.asof}.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    map_path = online_dir / f"public_feature_map_{args.asof}.csv"
    pd.DataFrame(mapping, columns=["pub_col", "source_col"]).to_csv(map_path, index=False)
    split_path = online_dir / f"split_info_{args.asof}.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print("Wrote:")
    print(f"  {train_path}  (rows={len(train_df)})")
    print(f"  {test_path}   (rows={len(test_df)})")
    print(f"  {map_path}    (rows={len(mapping)})")
    print(f"  {split_path}")
    print("Columns:", list(train_df.columns))
    print("Num public features:", train_df.shape[1] - 1, f"(mode={mode})")
    print("Split info:", split_info)


if __name__ == "__main__":
    main()

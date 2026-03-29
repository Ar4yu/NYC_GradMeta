import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from nyc_gradmeta.utils import online_artifact_stem


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


def load_observed_date_range(csv_path: Path, value_col: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError(f"{csv_path} must include a 'date' column.")
    if value_col not in df.columns:
        raise ValueError(f"{csv_path} must include '{value_col}'. Columns={df.columns.tolist()}")
    df["date"] = pd.to_datetime(df["date"])
    observed = df.dropna(subset=[value_col]).sort_values("date")
    if observed.empty:
        raise ValueError(f"{csv_path} has no observed rows for '{value_col}'.")
    return observed["date"].iloc[0], observed["date"].iloc[-1]


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
        "--smooth_cases_window",
        type=int,
        choices=[0, 3, 7],
        default=0,
        help="Apply a causal rolling mean to the target-only cases series. 0 disables smoothing.",
    )
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
        default=170,
        help="Use only the last N days up to ASOF before splitting. Defaults to 170 to match OpenTable coverage.",
    )
    ap.add_argument(
        "--matched_window_with_opentable",
        action="store_true",
        help=(
            "Use the true observed public/OpenTable overlap window for a fair A/B experiment. "
            "This clips the window to the joint overlap and ignores --window_days."
        ),
    )
    ap.add_argument(
        "--opentable_csv",
        default="data/processed/opentable_yoy_daily.csv",
        help="Processed OpenTable CSV used to define the matched overlap window.",
    )
    ap.add_argument(
        "--opentable_col",
        default="yoy_seated_diner",
        help="OpenTable column used to determine the observed overlap range.",
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

    # Build target columns from configured source column.
    src = nyc.get("target_source_column", "total_cases")
    if src not in df.columns:
        raise ValueError(f"Configured target_source_column '{src}' not found in master_daily_csv columns.")
    df["cases_raw"] = parse_numeric_column(df[src], src).astype(float)
    if args.smooth_cases_window > 0:
        df["cases"] = (
            df["cases_raw"].rolling(window=args.smooth_cases_window, min_periods=1).mean().astype(float)
        )
    else:
        df["cases"] = df["cases_raw"].astype(float)

    mode = nyc.get("public_feature_mode", "all_public")
    feature_df, mapping = build_public_features(
        df=df,
        mode=mode,
        target_col="cases",
        target_source_col=src,
        exclude_public_columns=(nyc.get("exclude_public_columns", []) + ["cases_raw", "date"]),
    )
    out = pd.concat(
        [
            df[["date", "cases_raw", "cases"]].reset_index(drop=True),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )
    out["cases_raw"] = out["cases_raw"].astype(float)
    out["cases"] = out["cases"].astype(float)
    if out.isna().any().any():
        raise ValueError("Prepared online dataframe contains NaNs.")

    # Restrict to <= ASOF before splitting.
    asof = pd.to_datetime(args.asof)

    actual_asof = asof
    matched_info: dict[str, object] = {
        "matched_window_with_opentable": bool(args.matched_window_with_opentable),
    }
    if args.matched_window_with_opentable:
        opentable_path = Path(args.opentable_csv)
        ot_start, ot_end = load_observed_date_range(opentable_path, args.opentable_col)
        public_start = df["date"].min()
        public_end = df["date"].max()
        joint_start = max(public_start, ot_start)
        joint_end = min(public_end, ot_end, asof)
        if joint_end < joint_start:
            raise ValueError(
                f"No joint public/OpenTable overlap for requested asof={args.asof}. "
                f"public=[{public_start.date()}, {public_end.date()}], "
                f"opentable=[{ot_start.date()}, {ot_end.date()}]."
            )
        actual_asof = joint_end
        matched_info.update(
            {
                "requested_asof": args.asof,
                "actual_asof": str(actual_asof.date()),
                "public_start": str(public_start.date()),
                "public_end": str(public_end.date()),
                "opentable_observed_start": str(ot_start.date()),
                "opentable_observed_end": str(ot_end.date()),
                "joint_start": str(joint_start.date()),
                "joint_end": str(joint_end.date()),
                "requested_window_days": int(args.window_days),
                "window_mode": "full_joint_overlap",
            }
        )

    # Restrict to <= asof
    mask_upto = df["date"] <= actual_asof
    df_upto = df.loc[mask_upto].copy()
    out_upto = out.loc[mask_upto].reset_index(drop=True)
    if df_upto.empty:
        raise ValueError(f"No rows found in master_daily_csv up to asof={args.asof}.")

    # Find index split based on date range
    dates_upto = df_upto["date"].reset_index(drop=True)
    diffs_upto = dates_upto.diff().dropna().dt.days
    if not (diffs_upto == 1).all():
        raise ValueError("Rows up to asof are not daily contiguous.")

    if args.matched_window_with_opentable:
        joint_start = pd.to_datetime(matched_info["joint_start"])
        joint_mask = dates_upto >= joint_start
        out_upto = out_upto.loc[joint_mask].reset_index(drop=True)
        dates_upto = dates_upto.loc[joint_mask].reset_index(drop=True)
        actual_window_days = int(len(out_upto))
    else:
        if args.window_days < 2:
            raise ValueError("--window_days must be >= 2.")
        start_idx = max(0, len(out_upto) - int(args.window_days))
        out_upto = out_upto.iloc[start_idx:].reset_index(drop=True)
        dates_upto = dates_upto.iloc[start_idx:].reset_index(drop=True)
        actual_window_days = int(len(out_upto))
    if len(out_upto) < 2:
        raise ValueError("Need at least 2 rows after ASOF/window filtering to split train/test.")

    if args.split_mode == "horizon":
        test_days = int(args.test_days) if args.test_days is not None else int(nyc["test_days"])
        if len(out_upto) <= test_days:
            raise ValueError(
                f"Need more than {test_days} rows in the experiment window; found {len(out_upto)}."
            )
        test_start = dates_upto.iloc[-test_days]
        test_end = dates_upto.iloc[-1]
        test_mask = dates_upto >= test_start
        if test_mask.sum() != test_days:
            raise ValueError(
                f"Expected exactly {test_days} test days between {test_start.date()} and {test_end.date()}, "
                f"found {int(test_mask.sum())}. Check missing dates in master_daily_csv."
            )
        train_df = out_upto.loc[~test_mask].reset_index(drop=True)
        test_df = out_upto.loc[test_mask].reset_index(drop=True)
        train_dates = dates_upto.loc[~test_mask].reset_index(drop=True)
        test_dates = dates_upto.loc[test_mask].reset_index(drop=True)
        split_info = {
            "split_mode": "horizon",
            "test_days": test_days,
            "train_days": int(len(train_df)),
            "window_days": actual_window_days,
            "smooth_cases_window": int(args.smooth_cases_window),
            "requested_asof": args.asof,
            "actual_asof": str(actual_asof.date()),
            "window_start": str(dates_upto.iloc[0].date()),
            "window_end": str(dates_upto.iloc[-1].date()),
            "train_start": str(train_dates.iloc[0].date()),
            "train_end": str(train_dates.iloc[-1].date()),
            "test_start": str(test_dates.iloc[0].date()),
            "test_end": str(test_dates.iloc[-1].date()),
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
        train_dates = dates_upto.iloc[:train_n].reset_index(drop=True)
        test_dates = dates_upto.iloc[train_n:].reset_index(drop=True)
        split_info = {
            "split_mode": "ratio",
            "train_ratio": ratio,
            "train_days": int(len(train_df)),
            "test_days": int(len(test_df)),
            "window_days": actual_window_days,
            "smooth_cases_window": int(args.smooth_cases_window),
            "requested_asof": args.asof,
            "actual_asof": str(actual_asof.date()),
            "window_start": str(dates_upto.iloc[0].date()),
            "window_end": str(dates_upto.iloc[-1].date()),
            "train_start": str(train_dates.iloc[0].date()),
            "train_end": str(train_dates.iloc[-1].date()),
            "test_start": str(test_dates.iloc[0].date()),
            "test_end": str(test_dates.iloc[-1].date()),
        }

    split_info.update(matched_info)

    # Save
    suffix_train = online_artifact_stem(
        "train",
        args.asof,
        args.window_days,
        args.smooth_cases_window,
        matched_window_with_opentable=args.matched_window_with_opentable,
    )
    suffix_test = online_artifact_stem(
        "test",
        args.asof,
        args.window_days,
        args.smooth_cases_window,
        matched_window_with_opentable=args.matched_window_with_opentable,
    )
    train_path = online_dir / f"{suffix_train}.csv"
    test_path = online_dir / f"{suffix_test}.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    map_suffix = online_artifact_stem(
        "public_feature_map",
        args.asof,
        args.window_days,
        args.smooth_cases_window,
        matched_window_with_opentable=args.matched_window_with_opentable,
    )
    map_path = online_dir / f"{map_suffix}.csv"
    pd.DataFrame(mapping, columns=["pub_col", "source_col"]).to_csv(map_path, index=False)
    split_suffix = online_artifact_stem(
        "split_info",
        args.asof,
        args.window_days,
        args.smooth_cases_window,
        matched_window_with_opentable=args.matched_window_with_opentable,
    )
    split_path = online_dir / f"{split_suffix}.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print("Wrote:")
    print(f"  {train_path}  (rows={len(train_df)})")
    print(f"  {test_path}   (rows={len(test_df)})")
    print(f"  {map_path}    (rows={len(mapping)})")
    print(f"  {split_path}")
    print("Columns:", list(train_df.columns))
    print("Num public features:", train_df.shape[1] - 3, f"(mode={mode})")
    print("Cases target:", "cases", f"({args.smooth_cases_window}-day causal rolling mean)" if args.smooth_cases_window else "(raw)")
    print("Split info:", split_info)


if __name__ == "__main__":
    main()

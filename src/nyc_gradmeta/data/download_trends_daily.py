from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from pytrends.request import TrendReq


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "trends_us_ny_daily.csv"


# -----------------------------
# Config
# -----------------------------
#GEO = "US-NY"  # New York state
# If you truly need NYC (not state), Google Trends does NOT support county/city in the same way as mobility.
# The link you used had US-NY-501 (NYC DMA). pytrends typically uses geo like "US-NY".
# We'll keep US-NY for stability/reproducibility.
GEO = "US-NY-501"  # NYC DMA

SLEEP_SECONDS = 1.0  # polite rate-limiting

# Choose ONE:
# A) keyword: e.g. "covid" or "covid 19"
# B) topic id: e.g. "/g/11j2cc_qll" (what you shared)
# Using the topic id is better because it is language/variant-stable.
KW_LIST = ["/g/11j2cc_qll"]  # COVID-19 topic (from your link)
COLNAME = "trend_covid_topic"


@dataclass
class Window:
    start: pd.Timestamp
    end: pd.Timestamp  # inclusive


def make_windows(start: str, end: str, window_days: int = 85, overlap_days: int = 7) -> List[Window]:
    """
    Build overlapping windows so pytrends returns DAILY resolution.
    Rule of thumb: <= 90 days gives daily in UI; we use 85 with overlap to be safe.
    """
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)

    windows: List[Window] = []
    cur = s
    step = pd.Timedelta(days=window_days - overlap_days)

    while cur <= e:
        w_end = min(cur + pd.Timedelta(days=window_days - 1), e)
        windows.append(Window(cur, w_end))
        cur = cur + step

    return windows


def fetch_window(pytrends: TrendReq, kw_list: List[str], w: Window, geo: str) -> pd.DataFrame:
    timeframe = f"{w.start.date()} {w.end.date()}"
    pytrends.build_payload(kw_list=kw_list, timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()

    if df is None or df.empty:
        return pd.DataFrame(columns=["date"] + kw_list)

    df = df.reset_index().rename(columns={"date": "date"})
    # Drop partial row if present
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    return df


def stitch_and_rescale(frames: List[pd.DataFrame], kw: str) -> pd.DataFrame:
    """
    Each window is scaled 0-100 *within that window*.
    We rescale sequentially using overlap ratios so the full daily series is comparable.
    """
    if not frames:
        raise RuntimeError("No frames fetched.")

    # Ensure sorted and clean
    cleaned = []
    for f in frames:
        if f.empty:
            continue
        f = f[["date", kw]].copy()
        f["date"] = pd.to_datetime(f["date"])
        f = f.dropna(subset=["date"])
        f = f.sort_values("date")
        cleaned.append(f)

    if not cleaned:
        raise RuntimeError("All fetched frames were empty. Try different keyword/topic or smaller range.")

    # Start with first window as baseline
    out = cleaned[0].copy()
    out["value"] = out[kw].astype(float)

    # Sequentially append, scaling each next window to match previous window in overlap
    for nxt in cleaned[1:]:
        nxt = nxt.copy()
        nxt["value"] = nxt[kw].astype(float)

        # Find overlap dates
        overlap = pd.merge(
            out[["date", "value"]],
            nxt[["date", "value"]],
            on="date",
            how="inner",
            suffixes=("_prev", "_nxt"),
        )

        # Compute scale factor using robust statistic (median of ratios) over nonzero overlap
        overlap = overlap[(overlap["value_nxt"] > 0) & (overlap["value_prev"] > 0)]
        if overlap.empty:
            # If no nonzero overlap, fallback to using max alignment if possible
            prev_max = out["value"].max()
            nxt_max = nxt["value"].max()
            scale = (prev_max / nxt_max) if nxt_max > 0 else 1.0
        else:
            ratios = overlap["value_prev"] / overlap["value_nxt"]
            scale = float(ratios.median())

        # Apply scaling to next window values
        nxt["value"] = nxt["value"] * scale

        # Append only new dates (avoid duplicates in overlap)
        nxt_new = nxt[~nxt["date"].isin(out["date"])]

        out = pd.concat([out, nxt_new[["date", "value"]]], ignore_index=True)

    out = out.sort_values("date").reset_index(drop=True)

    # Optional: normalize to 0-100 over the whole period (common for ML features)
    maxv = out["value"].max()
    if maxv > 0:
        out["value_norm_0_100"] = out["value"] / maxv * 100.0
    else:
        out["value_norm_0_100"] = out["value"]

    return out[["date", "value_norm_0_100"]].rename(columns={"value_norm_0_100": COLNAME})


def main(
    start: str = "2020-01-01",
    end: str = "2022-12-31",
    geo: str = GEO,
    kw_list: Optional[List[str]] = None,
):
    kw_list = kw_list or KW_LIST
    if len(kw_list) != 1:
        raise ValueError("This script expects exactly 1 keyword/topic for now (simpler scaling).")

    kw = kw_list[0]

    windows = make_windows(start, end, window_days=85, overlap_days=7)
    print(f"Downloading daily windows: {len(windows)} windows from {start} to {end} geo={geo} kw={kw}")

    pytrends = TrendReq(hl="en-US", tz=360)  # tz minutes offset; 360 = UTC-6 (doesn't matter much for daily)

    frames: List[pd.DataFrame] = []
    for i, w in enumerate(windows, 1):
        print(f"[{i}/{len(windows)}] {w.start.date()} → {w.end.date()}")
        df = fetch_window(pytrends, kw_list, w, geo)
        print("  rows:", len(df), "min:", df["date"].min() if not df.empty else None, "max:", df["date"].max() if not df.empty else None)
        frames.append(df)
        time.sleep(SLEEP_SECONDS)

    stitched = stitch_and_rescale(frames, kw=kw)

    # Ensure full continuous daily coverage (some days might be missing if Google returns sparse)
    stitched = stitched.sort_values("date")
    full = pd.date_range(stitched["date"].min(), stitched["date"].max(), freq="D")
    stitched = stitched.set_index("date").reindex(full).rename_axis("date").reset_index()

    # Interpolate small gaps if any
    stitched[COLNAME] = stitched[COLNAME].interpolate(limit_direction="both")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stitched.to_csv(OUT_PATH, index=False)

    print("\nSaved:", OUT_PATH)
    print("Shape:", stitched.shape)
    print(stitched.head())
    print(stitched.tail())

    # quick missingness check
    missing = stitched[COLNAME].isna().sum()
    print("Missing values after interpolate:", missing)


if __name__ == "__main__":
    main()

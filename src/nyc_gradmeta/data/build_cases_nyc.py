import pandas as pd
from pathlib import Path

# -------------------------------------------------
# Paths
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "COVID-19_Daily_Counts_of_Cases,_Hospitalizations,_and_Deaths_20260213.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "cases_nyc_daily.csv"

# -------------------------------------------------
# Load
# -------------------------------------------------
df = pd.read_csv(RAW_PATH)

# -------------------------------------------------
# Select only citywide columns
# -------------------------------------------------
df = df[[
    "date_of_interest",
    "CASE_COUNT",
    "PROBABLE_CASE_COUNT",
    "HOSPITALIZED_COUNT",
    "DEATH_COUNT",
]].copy()

# -------------------------------------------------
# Clean dates
# -------------------------------------------------
df["date"] = pd.to_datetime(df["date_of_interest"], errors="coerce")
df = df.dropna(subset=["date"])

# -------------------------------------------------
# Rename for consistency
# -------------------------------------------------
df = df.rename(columns={
    "CASE_COUNT": "cases",
    "PROBABLE_CASE_COUNT": "probable_cases",
    "HOSPITALIZED_COUNT": "hospitalizations",
    "DEATH_COUNT": "deaths",
})

# Drop original date column
df = df.drop(columns=["date_of_interest"])

# -------------------------------------------------
# Sort + ensure daily continuity
# -------------------------------------------------
df = df.sort_values("date")

full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
df = df.set_index("date").reindex(full_range).rename_axis("date").reset_index()

# Fill missing days with 0 (safe for counts)
for col in ["cases", "probable_cases", "hospitalizations", "deaths"]:
    df[col] = df[col].fillna(0)

# -------------------------------------------------
# Save
# -------------------------------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Shape:", df.shape)
print(df.head())
print(df.tail())

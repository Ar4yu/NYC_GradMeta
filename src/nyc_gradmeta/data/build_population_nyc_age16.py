from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_population_nyc_age16_2020() -> pd.DataFrame:
    """
    NYC population by 16 age bins (Prem-style bins):
      0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39,
      40-44, 45-49, 50-54, 55-59, 60-64, 65-69, 70-74, 75+

    Source (NYC, 2020): NYS DOH Vital Statistics Table 01 (NYC section).
    https://health.ny.gov/statistics/vital_statistics/2020/table01.htm

    Notes on aggregation from the source table:
      - 0-4 = (<1) + (1) + (2) + (3) + (4)
      - 15-19 = (15-17) + (18-19)
      - 75+ = (75-79) + (80-84) + (85+)
    """
    # Raw (Total column) values from the NYC 2020 table
    raw = {
        "<1": 106_234,
        "1": 102_733,
        "2": 102_040,
        "3": 100_502,
        "4": 99_463,
        "5-9": 482_295,
        "10-14": 438_717,
        "15-17": 261_919,
        "18-19": 169_958,
        "20-24": 501_041,
        "25-29": 727_506,
        "30-34": 728_639,
        "35-39": 602_853,
        "40-44": 524_865,
        "45-49": 497_625,
        "50-54": 505_869,
        "55-59": 511_981,
        "60-64": 480_222,
        "65-69": 403_354,
        "70-74": 333_148,
        "75-79": 227_638,
        "80-84": 163_320,
        "85+": 181_291,
        "Total": 8_253_213,
    }

    pop16 = {
        "0-4": raw["<1"] + raw["1"] + raw["2"] + raw["3"] + raw["4"],
        "5-9": raw["5-9"],
        "10-14": raw["10-14"],
        "15-19": raw["15-17"] + raw["18-19"],
        "20-24": raw["20-24"],
        "25-29": raw["25-29"],
        "30-34": raw["30-34"],
        "35-39": raw["35-39"],
        "40-44": raw["40-44"],
        "45-49": raw["45-49"],
        "50-54": raw["50-54"],
        "55-59": raw["55-59"],
        "60-64": raw["60-64"],
        "65-69": raw["65-69"],
        "70-74": raw["70-74"],
        "75+": raw["75-79"] + raw["80-84"] + raw["85+"],
    }

    df = pd.DataFrame(
        {"age_group": list(pop16.keys()), "population": list(pop16.values())}
    )

    total_16 = int(df["population"].sum())
    if total_16 != int(raw["Total"]):
        raise RuntimeError(
            f"Sanity check failed: sum(pop16)={total_16} but raw Total={raw['Total']}"
        )

    return df


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    # you said: “put them under data processed no more sub directories”
    out = project_root / "data" / "processed" / "population_nyc_age16_2020.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = build_population_nyc_age16_2020()
    df.to_csv(out, index=False)

    print("Saved:", out)
    print("Rows:", len(df))
    print("Total population:", int(df["population"].sum()))
    print(df)


if __name__ == "__main__":
    main()

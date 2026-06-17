from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

ZORI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zori/"
    "Metro_zori_uc_sfrcondomfr_sm_month.csv?t=1781532004"
)


def _minmax(series):
    spread = series.max() - series.min()
    if pd.isna(spread) or spread == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / spread


def download_zori():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ZORI_URL)
    out_path = RAW_DIR / "zillow_zori_metro.csv"
    df.to_csv(out_path, index=False)
    return df


def build_latest_market_table(df):
    date_cols = [col for col in df.columns if col[:4].isdigit() and "-" in col]
    latest_col = date_cols[-1]
    prior_col = date_cols[-13] if len(date_cols) >= 13 else date_cols[0]

    table = df[["RegionID", "SizeRank", "RegionName", "RegionType", "StateName", latest_col, prior_col]].copy()
    table = table.rename(
        columns={
            latest_col: "observed_rent",
            prior_col: "rent_12m_ago",
        }
    )
    table["latest_date"] = latest_col
    table["prior_year_date"] = prior_col
    table["observed_rent"] = pd.to_numeric(table["observed_rent"], errors="coerce")
    table["rent_12m_ago"] = pd.to_numeric(table["rent_12m_ago"], errors="coerce")
    table = table.dropna(subset=["observed_rent", "rent_12m_ago"])
    table["yoy_growth_pct"] = ((table["observed_rent"] - table["rent_12m_ago"]) / table["rent_12m_ago"] * 100.0)
    table["rent_rank"] = table["observed_rent"].rank(ascending=False, method="dense").astype(int)

    table["market_pressure_score"] = (
        0.55 * _minmax(table["observed_rent"])
        + 0.45 * _minmax(table["yoy_growth_pct"])
    ).replace([np.inf, -np.inf], 0).fillna(0)
    return table.sort_values("market_pressure_score", ascending=False)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = download_zori()
    table = build_latest_market_table(raw)
    out_path = PROCESSED_DIR / "zillow_zori_metro_latest.csv"
    table.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(table)} metro rows")


if __name__ == "__main__":
    main()

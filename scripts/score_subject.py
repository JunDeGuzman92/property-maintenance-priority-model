import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from market_comp import analyze_market_frame
from download_data import build_latest_market_table, download_zori


def main():
    data_path = ROOT / "data" / "processed" / "zillow_zori_metro_latest.csv"
    if data_path.exists():
        market_df = pd.read_csv(data_path)
    else:
        market_df = build_latest_market_table(download_zori())
    estimates = analyze_market_frame(market_df)
    cols = [
        "RegionName",
        "StateName",
        "latest_date",
        "observed_rent",
        "yoy_growth_pct",
        "market_pressure_score",
        "review_note",
    ]
    print(json.dumps(estimates[cols].head(10).to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()

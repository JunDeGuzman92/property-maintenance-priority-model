from pathlib import Path
from urllib.parse import urlencode

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

NYC_311_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"


def _read_socrata(resource_url, params):
    url = f"{resource_url}?{urlencode(params)}"
    return pd.read_json(url)


def download_public_requests(limit=50000):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    params = {
        "$select": (
            "unique_key,created_date,closed_date,complaint_type,descriptor,"
            "borough,incident_zip,status,location_type,open_data_channel_type"
        ),
        "$where": "agency='HPD' AND created_date >= '2025-01-01T00:00:00' AND complaint_type IS NOT NULL",
        "$limit": str(limit),
    }
    df = _read_socrata(NYC_311_URL, params)
    df.to_csv(RAW_DIR / "nyc_311_hpd_housing_requests.csv", index=False)
    return df


def build_training_table(df):
    table = df.copy()
    table["created_date"] = pd.to_datetime(table["created_date"], errors="coerce")
    table["closed_date"] = pd.to_datetime(table["closed_date"], errors="coerce")
    table = table.dropna(subset=["created_date", "complaint_type"])
    closure_hours = (table["closed_date"] - table["created_date"]).dt.total_seconds() / 3600.0
    table["closure_days"] = closure_hours / 24.0
    table["is_open"] = table["closed_date"].isna() | ~table["status"].astype(str).str.upper().eq("CLOSED")
    table["created_month"] = table["created_date"].dt.month
    table["is_winter"] = table["created_month"].isin([11, 12, 1, 2, 3]).astype(int)
    table["has_descriptor"] = table["descriptor"].notna().astype(int)

    urgent_complaints = {
        "HEAT/HOT WATER",
        "PLUMBING",
        "ELECTRIC",
        "WATER LEAK",
        "UNSANITARY CONDITION",
    }
    table["priority_followup"] = (
        table["is_open"]
        | (table["closure_days"].fillna(999) >= 7)
        | (
            table["complaint_type"].astype(str).str.upper().isin(urgent_complaints)
            & (table["closure_days"].fillna(999) >= 2)
        )
    ).astype(int)
    keep = [
        "unique_key",
        "created_month",
        "is_winter",
        "complaint_type",
        "descriptor",
        "has_descriptor",
        "borough",
        "incident_zip",
        "status",
        "location_type",
        "open_data_channel_type",
        "closure_days",
        "priority_followup",
    ]
    return table[keep].copy()


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = download_public_requests()
    table = build_training_table(raw)
    out_path = PROCESSED_DIR / "nyc_311_hpd_priority_training.csv"
    table.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(table)} service-request rows")


if __name__ == "__main__":
    main()

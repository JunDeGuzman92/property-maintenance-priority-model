import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from market_comp import run_market_analysis
from download_data import build_latest_market_table, download_zori


def main():
    data_path = ROOT / "data" / "processed" / "zillow_zori_metro_latest.csv"
    if not data_path.exists():
        raw = download_zori()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        build_latest_market_table(raw).to_csv(data_path, index=False)
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "market_rent_recommendations.csv"
    estimates = run_market_analysis(data_path, out_path)
    print(estimates.head(25).to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

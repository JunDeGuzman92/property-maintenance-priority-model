import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from maintenance_priority import LogisticPriorityModel, binary_metrics, build_feature_matrix
from download_data import build_training_table, download_public_requests


TARGET = "priority_followup"


def load_training_data():
    data_path = ROOT / "data" / "processed" / "nyc_311_hpd_priority_training.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    raw = download_public_requests()
    table = build_training_table(raw)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(data_path, index=False)
    return table


def main():
    df = load_training_data()
    features = build_feature_matrix(df)
    y = df[TARGET].astype(int)
    test_mask = (df.index % 4) == 0

    model = LogisticPriorityModel()
    model.fit(features.loc[~test_mask].to_numpy(), y.loc[~test_mask].to_numpy())
    preds = model.predict(features.loc[test_mask].to_numpy())
    metrics = binary_metrics(y.loc[test_mask].to_numpy(), preds)

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "maintenance_priority_model.json"
    out_path.write_text(
        json.dumps(model.to_dict(features.columns, metrics), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

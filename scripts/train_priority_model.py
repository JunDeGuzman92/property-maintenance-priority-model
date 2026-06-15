import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maintenance_priority import LogisticPriorityModel, binary_metrics, build_feature_matrix


def main():
    df = pd.read_csv(ROOT / "data" / "synthetic_work_orders.csv")
    features = build_feature_matrix(df)
    y = df["urgent_followup"].astype(int)
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


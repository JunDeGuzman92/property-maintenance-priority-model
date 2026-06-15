import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maintenance_priority import LogisticPriorityModel, build_feature_matrix, load_model, score_work_order


class MaintenancePriorityTests(unittest.TestCase):
    def test_feature_matrix_has_category_columns(self):
        df = pd.read_csv(ROOT / "data" / "synthetic_work_orders.csv")
        features = build_feature_matrix(df)
        self.assertIn("category_hvac", features.columns)
        self.assertIn("safety_flag", features.columns)

    def test_model_predicts_probabilities(self):
        df = pd.read_csv(ROOT / "data" / "synthetic_work_orders.csv")
        x = build_feature_matrix(df).to_numpy()
        y = df["urgent_followup"].to_numpy()
        model = LogisticPriorityModel(epochs=200)
        model.fit(x, y)
        probs = model.predict_proba(x[:3])
        self.assertEqual(len(probs), 3)
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())

    def test_inference_returns_priority_score(self):
        payload = load_model(ROOT / "artifacts" / "maintenance_priority_model.json")
        result = score_work_order(
            {
                "age_days": 4,
                "category": "hvac",
                "occupied_unit": 1,
                "safety_flag": 1,
                "recurrence_count": 2,
                "asset_age_years": 14,
                "after_hours": 1,
            },
            model_payload=payload,
        )
        self.assertIn("urgent_followup_score", result)
        self.assertIn(result["review_label"], {"urgent_review", "standard_queue"})


if __name__ == "__main__":
    unittest.main()

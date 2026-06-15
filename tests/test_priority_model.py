import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maintenance_priority import LogisticPriorityModel, build_feature_matrix


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


if __name__ == "__main__":
    unittest.main()


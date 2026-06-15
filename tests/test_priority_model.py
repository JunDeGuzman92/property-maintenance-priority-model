import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maintenance_priority import LogisticPriorityModel, build_feature_matrix, load_model, score_work_order


class MaintenancePriorityTests(unittest.TestCase):
    def test_feature_matrix_has_category_columns(self):
        df = pd.DataFrame(
            {
                "created_month": [1, 7],
                "is_winter": [1, 0],
                "complaint_type": ["HEAT/HOT WATER", "PLUMBING"],
                "has_descriptor": [1, 1],
                "borough": ["BRONX", "BROOKLYN"],
                "location_type": ["Residential Building", "Apartment"],
                "open_data_channel_type": ["PHONE", "ONLINE"],
            }
        )
        features = build_feature_matrix(df)
        self.assertIn("complaint_heat_hot_water", features.columns)
        self.assertIn("borough_bronx", features.columns)

    def test_model_predicts_probabilities(self):
        df = pd.DataFrame(
            {
                "created_month": [1, 7, 2, 9, 12, 5],
                "is_winter": [1, 0, 1, 0, 1, 0],
                "complaint_type": ["HEAT/HOT WATER", "PLUMBING", "ELECTRIC", "GENERAL", "WATER LEAK", "APPLIANCE"],
                "has_descriptor": [1, 1, 0, 1, 1, 0],
                "borough": ["BRONX", "BROOKLYN", "QUEENS", "MANHATTAN", "BRONX", "BROOKLYN"],
                "location_type": ["Residential Building"] * 6,
                "open_data_channel_type": ["PHONE", "ONLINE", "MOBILE", "PHONE", "ONLINE", "PHONE"],
            }
        )
        x = build_feature_matrix(df).to_numpy()
        y = [1, 0, 1, 0, 1, 0]
        model = LogisticPriorityModel(epochs=200)
        model.fit(x, y)
        probs = model.predict_proba(x[:3])
        self.assertEqual(len(probs), 3)
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())

    def test_inference_returns_priority_score(self):
        payload = load_model(ROOT / "artifacts" / "maintenance_priority_model.json")
        result = score_work_order(
            {
                "created_month": 1,
                "is_winter": 1,
                "complaint_type": "HEAT/HOT WATER",
                "has_descriptor": 1,
                "borough": "BRONX",
                "location_type": "Residential Building",
                "open_data_channel_type": "PHONE",
            },
            model_payload=payload,
        )
        self.assertIn("urgent_followup_score", result)
        self.assertIn(result["review_label"], {"urgent_review", "standard_queue"})


if __name__ == "__main__":
    unittest.main()

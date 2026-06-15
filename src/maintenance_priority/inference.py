import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_feature_matrix


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "maintenance_priority_model.json"


def load_model(path=DEFAULT_MODEL_PATH):
    """Load a serialized maintenance-priority model artifact."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sigmoid(value):
    value = np.clip(value, -30, 30)
    return 1.0 / (1.0 + np.exp(-value))


def _top_drivers(values, feature_names, weights, limit=3):
    contributions = values.reshape(-1) * weights
    ranked = sorted(
        zip(feature_names, contributions),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return [
        {"feature": name, "contribution": round(float(contribution), 4)}
        for name, contribution in ranked[:limit]
    ]


def score_work_order(record, model_payload=None, threshold=0.5):
    """Score one work order for urgent follow-up review."""
    payload = model_payload or load_model()
    feature_names = payload["feature_names"]
    features = build_feature_matrix(pd.DataFrame([record]))
    missing = [name for name in feature_names if name not in features.columns]
    if missing:
        raise ValueError(f"Missing required feature column(s): {', '.join(missing)}")
    raw = features[feature_names].to_numpy(dtype=float)
    mean = np.array(payload["mean"], dtype=float)
    scale = np.array(payload["scale"], dtype=float)
    weights = np.array(payload["weights"], dtype=float)
    scaled = (raw - mean) / scale
    probability = float(_sigmoid((scaled @ weights) + float(payload["bias"]))[0])
    return {
        "urgent_followup_score": round(probability, 4),
        "review_label": "urgent_review" if probability >= threshold else "standard_queue",
        "threshold": threshold,
        "top_drivers": _top_drivers(scaled, feature_names, weights),
    }

import numpy as np
import pandas as pd

from maintenance_priority.features import build_feature_matrix


def _sigmoid(values):
    values = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-values))


def _payload_arrays(model_payload):
    feature_names = list(model_payload.get("feature_names", []))
    weights = np.asarray(model_payload.get("weights", []), dtype=float)
    mean = np.asarray(model_payload.get("mean", []), dtype=float)
    scale = np.asarray(model_payload.get("scale", []), dtype=float)
    bias = float(model_payload.get("bias", 0.0))
    return feature_names, weights, mean, scale, bias


def _business_feature_name(name):
    readable = name.replace("_", " ")
    replacements = {
        "complaint ": "Complaint: ",
        "borough ": "Borough: ",
        "channel ": "Channel: ",
        "created month": "Created month",
        "is winter": "Winter request",
        "has descriptor": "Has descriptor",
        "is residential location": "Residential location",
    }
    for old, new in replacements.items():
        if readable.startswith(old):
            readable = readable.replace(old, new, 1)
            break
    return readable.title() if ":" not in readable else readable.title().replace(": ", ": ")


def align_feature_matrix(df, model_payload):
    """Build model features and align them to a saved artifact without crashing."""
    feature_names, weights, mean, scale, _ = _payload_arrays(model_payload)
    warnings = []

    if not feature_names:
        return pd.DataFrame(index=df.index), ["The model artifact does not list feature names."]

    features = build_feature_matrix(df)
    missing_features = [name for name in feature_names if name not in features.columns]
    for name in missing_features:
        features[name] = 0.0

    if missing_features:
        warnings.append(
            "Model feature columns were missing and filled with 0: "
            + ", ".join(missing_features)
        )

    if len(weights) != len(feature_names) or len(mean) != len(feature_names) or len(scale) != len(feature_names):
        warnings.append("The model artifact arrays do not match the feature list length.")
        return pd.DataFrame(index=df.index), warnings

    aligned = features[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return aligned, warnings


def score_dataframe(df, model_payload, threshold=0.5):
    """Vectorized scoring for dashboard-sized dataframes."""
    scored = df.copy()
    feature_names, weights, mean, scale, bias = _payload_arrays(model_payload)
    features, warnings = align_feature_matrix(scored, model_payload)

    if features.empty and len(scored) > 0:
        scored["priority_score"] = np.nan
        scored["review_label"] = "model_unavailable"
        return scored, warnings

    raw = features.to_numpy(dtype=float)
    scaled = (raw - mean) / scale
    probabilities = _sigmoid((scaled @ weights) + bias)
    scored["priority_score"] = np.round(probabilities, 4)
    scored["review_label"] = np.where(
        probabilities >= threshold,
        "urgent_review",
        "standard_queue",
    )
    return scored, warnings


def contribution_summary(df, model_payload, limit=15):
    """Summarize average absolute linear-model contributions by feature."""
    feature_names, weights, mean, scale, _ = _payload_arrays(model_payload)
    features, warnings = align_feature_matrix(df, model_payload)
    if features.empty:
        return pd.DataFrame(), warnings

    scaled = (features.to_numpy(dtype=float) - mean) / scale
    contributions = scaled * weights
    summary = pd.DataFrame(
        {
            "feature": feature_names,
            "business_label": [_business_feature_name(name) for name in feature_names],
            "mean_contribution": contributions.mean(axis=0),
            "mean_abs_contribution": np.abs(contributions).mean(axis=0),
        }
    )
    summary["direction"] = np.where(summary["mean_contribution"] >= 0, "Raises score", "Lowers score")
    return summary.sort_values("mean_abs_contribution", ascending=False).head(limit), warnings


def row_contributions(row, model_payload, limit=6):
    """Return the strongest model drivers for one row."""
    feature_names, weights, mean, scale, _ = _payload_arrays(model_payload)
    features, warnings = align_feature_matrix(pd.DataFrame([row]), model_payload)
    if features.empty:
        return pd.DataFrame(), warnings

    scaled = (features.to_numpy(dtype=float) - mean) / scale
    contributions = scaled.reshape(-1) * weights
    drivers = pd.DataFrame(
        {
            "feature": feature_names,
            "business_label": [_business_feature_name(name) for name in feature_names],
            "contribution": contributions,
            "abs_contribution": np.abs(contributions),
        }
    )
    return drivers.sort_values("abs_contribution", ascending=False).head(limit), warnings


def optional_shap_summary(df, model_payload, max_rows=120, background_rows=40):
    """Compute a small SHAP summary when the optional package is available."""
    try:
        import shap
    except ImportError:
        return pd.DataFrame(), "Install shap to enable sample-based SHAP explanations."

    feature_names, weights, mean, scale, bias = _payload_arrays(model_payload)
    features, warnings = align_feature_matrix(df, model_payload)
    if warnings:
        return pd.DataFrame(), " ".join(warnings)
    if features.empty:
        return pd.DataFrame(), "No rows are available for SHAP explanations."

    sample = features.head(max_rows)
    background = features.head(min(background_rows, len(features)))

    def predict_fn(values):
        values = np.asarray(values, dtype=float)
        scaled = (values - mean) / scale
        return _sigmoid((scaled @ weights) + bias)

    try:
        explainer = shap.Explainer(predict_fn, background.to_numpy(dtype=float))
        explanation = explainer(sample.to_numpy(dtype=float))
        values = np.asarray(explanation.values)
    except Exception as exc:
        return pd.DataFrame(), f"SHAP could not run for this dataset: {exc}"

    if values.ndim == 3:
        values = values[:, :, 0]

    summary = pd.DataFrame(
        {
            "feature": feature_names,
            "business_label": [_business_feature_name(name) for name in feature_names],
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }
    )
    return summary.sort_values("mean_abs_shap", ascending=False), ""

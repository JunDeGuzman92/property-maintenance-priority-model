import pandas as pd


CATEGORIES = ["appliance", "cosmetic", "electrical", "hvac", "life_safety", "plumbing"]


def build_feature_matrix(df):
    """Convert work-order rows into a numeric ML feature matrix."""
    features = pd.DataFrame(index=df.index)
    numeric_cols = [
        "age_days",
        "occupied_unit",
        "safety_flag",
        "recurrence_count",
        "asset_age_years",
        "after_hours",
    ]
    for col in numeric_cols:
        features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for cat in CATEGORIES:
        features[f"category_{cat}"] = (df["category"].astype(str) == cat).astype(int)
    return features


import numpy as np
import pandas as pd


def _minmax(series):
    series = pd.to_numeric(series, errors="coerce")
    spread = series.max() - series.min()
    if pd.isna(spread) or spread == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / spread


def score_market_trends(df, review_growth_pct=4.0):
    """Score public market-rent trend rows for review."""
    scored = df.copy()
    scored["observed_rent"] = pd.to_numeric(scored["observed_rent"], errors="coerce")
    scored["yoy_growth_pct"] = pd.to_numeric(scored["yoy_growth_pct"], errors="coerce")

    rent_norm = _minmax(scored["observed_rent"])
    growth_norm = _minmax(scored["yoy_growth_pct"])
    scored["market_pressure_score"] = (
        0.55 * rent_norm.fillna(0) + 0.45 * growth_norm.fillna(0)
    ).round(4)
    high_growth = scored["yoy_growth_pct"] >= review_growth_pct
    scored["review_note"] = np.where(high_growth, "review elevated rent growth", "within review band")
    return scored.sort_values("market_pressure_score", ascending=False)


def _safe_exp_decay(values, scale):
    values = np.asarray(values, dtype=float)
    return np.exp(-np.maximum(values, 0) / scale)


def score_comps(df):
    """Score comparable units using transparent business rules."""
    scored = df.copy()
    sqft_delta = (scored["comp_sqft"] - scored["subject_sqft"]).abs()
    scored["sqft_similarity"] = np.exp(-sqft_delta / 150.0)
    scored["distance_score"] = _safe_exp_decay(scored["distance_km"], 4.0)
    scored["recency_score"] = _safe_exp_decay(scored["days_since_observed"], 60.0)
    scored["type_match"] = (scored["subject_unit_type"] == scored["comp_unit_type"]).astype(float)
    scored["comp_score"] = (
        0.30 * scored["type_match"]
        + 0.25 * scored["sqft_similarity"]
        + 0.20 * scored["distance_score"]
        + 0.15 * scored["amenity_overlap"]
        + 0.10 * scored["recency_score"]
    )
    return scored


def estimate_market_rent(scored):
    """Return one market-rent estimate per subject unit."""
    rows = []
    for subject_id, group in scored.groupby("subject_id"):
        weights = group["comp_score"].clip(lower=0.001)
        estimate = float(np.average(group["comp_rent"], weights=weights))
        current = float(group["subject_current_rent"].iloc[0])
        gap_pct = (estimate - current) / current * 100.0
        top_comp = group.sort_values("comp_score", ascending=False)["comp_id"].iloc[0]
        rows.append(
            {
                "subject_id": subject_id,
                "subject_unit_type": group["subject_unit_type"].iloc[0],
                "subject_sqft": int(group["subject_sqft"].iloc[0]),
                "current_rent": round(current, 2),
                "estimated_market_rent": round(estimate, 2),
                "market_gap_pct": round(gap_pct, 2),
                "comp_count": int(len(group)),
                "top_comp": top_comp,
            }
        )
    return pd.DataFrame(rows).sort_values("subject_id")

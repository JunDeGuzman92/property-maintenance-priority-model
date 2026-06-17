from pathlib import Path

import pandas as pd

from .scoring import estimate_market_rent, score_comps, score_market_trends


def analyze_market_frame(df, review_threshold_pct=4.0):
    """Run the full comparable-rent analysis on an in-memory DataFrame."""
    if {"observed_rent", "yoy_growth_pct"}.issubset(df.columns):
        return score_market_trends(df, review_growth_pct=review_threshold_pct)
    scored = score_comps(df)
    estimates = estimate_market_rent(scored)
    estimates["review_note"] = estimates["market_gap_pct"].apply(
        lambda value: "review pricing gap"
        if abs(value) >= review_threshold_pct
        else "within review band"
    )
    return estimates


def run_market_analysis(input_path, output_path, review_threshold_pct=4.0):
    """Run the comparable-rent pipeline from CSV input to CSV output."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = pd.read_csv(input_path)
    estimates = analyze_market_frame(df, review_threshold_pct=review_threshold_pct)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(output_path, index=False)
    return estimates

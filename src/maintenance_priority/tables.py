"""Comparison tables for the rent-comparison dashboard.

These builders turn the analysed DataFrame into clean, business-labelled
tables the leasing team can read at a glance or export. They expect the rent
gap columns (``rent_gap_pct``, ``rent_position``, ``suggested_action``) to be
present already - the app adds those once via ``market_comp.add_rent_gap`` so
the math lives in a single place.

Everything here is plain pandas so it can be unit-tested without Streamlit.
"""

from __future__ import annotations

import pandas as pd

from market_comp.analysis import MarketSchema


def competitor_ranking(df: pd.DataFrame, schema: MarketSchema, top_n: int = 20) -> pd.DataFrame:
    """Rank comparable markets/properties from most to least expensive.

    Returns a tidy, business-labelled table with the rent, gap to market, and
    suggested action for each comp - the kind of view that anchors a pricing
    discussion.
    """
    ranked = df.sort_values(schema.rent, ascending=False).head(top_n).copy()

    table = pd.DataFrame()
    table["Market / Property"] = ranked[schema.name].astype(str)
    if schema.state:
        table["State"] = ranked[schema.state].astype(str)
    if schema.unit_type:
        table["Unit Type"] = ranked[schema.unit_type].astype(str)
    table["Current Rent"] = pd.to_numeric(ranked[schema.rent], errors="coerce")
    if schema.yoy_growth:
        table["YoY Growth %"] = pd.to_numeric(ranked[schema.yoy_growth], errors="coerce")
    if schema.pressure_score:
        table["Market Pressure"] = pd.to_numeric(ranked[schema.pressure_score], errors="coerce")
    table["Rent Gap %"] = ranked["rent_gap_pct"]
    table["Position"] = ranked["rent_position"]
    table["Suggested Action"] = ranked["suggested_action"]

    return table.reset_index(drop=True)


def executive_summary(df: pd.DataFrame, schema: MarketSchema, market_median: float) -> pd.DataFrame:
    """The one-table summary leadership asks for.

    Columns follow the brief: property / market name, unit type, current rent,
    market median, rent gap, and a suggested action.
    """
    summary = pd.DataFrame()
    summary["Property / Market"] = df[schema.name].astype(str)
    summary["Unit Type"] = df[schema.unit_type].astype(str) if schema.unit_type else "All units"
    summary["Current Rent"] = pd.to_numeric(df[schema.rent], errors="coerce")
    summary["Market Median"] = market_median
    summary["Rent Gap %"] = df["rent_gap_pct"]
    summary["Position"] = df["rent_position"]
    summary["Suggested Action"] = df["suggested_action"]

    return summary.reset_index(drop=True)


def position_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Small count table: how many comps sit below / near / above market."""
    order = ["Below market", "Near market", "Above market"]
    counts = df["rent_position"].value_counts().reindex(order).fillna(0).astype(int)
    share = (counts / counts.sum() * 100).round(1) if counts.sum() else counts
    return pd.DataFrame(
        {
            "Position": counts.index,
            "Markets": counts.values,
            "Share %": share.values,
        }
    )

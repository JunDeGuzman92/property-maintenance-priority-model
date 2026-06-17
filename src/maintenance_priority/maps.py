"""Map visuals for the rent-comparison dashboard.

The public benchmark data carries a real location field - the two-letter state
code - but no street coordinates, so the default map is a US state choropleth
of median rent. If a richer internal file with latitude/longitude is loaded,
the app switches to a point map automatically.

Plotly's built-in geo maps are used throughout: they render offline and need
no map token or extra dependency beyond Plotly itself.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from market_comp.analysis import MarketSchema

INK = "#1f2937"


def choose_map(schema: MarketSchema) -> str | None:
    """Decide which map (if any) the data can support.

    Returns ``"points"`` when we have coordinates, ``"states"`` when we only
    have a state field, and ``None`` when there is nothing to map.
    """
    if schema.latitude and schema.longitude:
        return "points"
    if schema.state:
        return "states"
    return None


def state_rent_choropleth(df: pd.DataFrame, state_col: str, rent_col: str) -> go.Figure:
    """Median rent by US state - a quick read on where rents run hottest."""
    by_state = (
        df.assign(_rent=pd.to_numeric(df[rent_col], errors="coerce"))
        .dropna(subset=["_rent"])
    )
    # Keep clean two-letter state codes only.
    by_state = by_state[by_state[state_col].astype(str).str.strip().str.len() == 2]
    state_median = (
        by_state.groupby(by_state[state_col].astype(str).str.strip())["_rent"]
        .median()
        .reset_index()
    )
    state_median.columns = ["state", "median_rent"]

    fig = go.Figure(
        go.Choropleth(
            locations=state_median["state"],
            z=state_median["median_rent"],
            locationmode="USA-states",
            colorscale="Blues",
            colorbar=dict(title="Median rent", tickprefix="$"),
            hovertemplate="%{location}<br>Median rent $%{z:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Median market rent by state",
        geo=dict(scope="usa", lakecolor="white"),
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", color=INK),
        margin=dict(l=10, r=10, t=60, b=10),
        height=460,
    )
    return fig


def property_point_map(df: pd.DataFrame, schema: MarketSchema) -> go.Figure:
    """Plot each property as a point, coloured by rent (needs lat/long)."""
    points = df.copy()
    points["_lat"] = pd.to_numeric(points[schema.latitude], errors="coerce")
    points["_lon"] = pd.to_numeric(points[schema.longitude], errors="coerce")
    points["_rent"] = pd.to_numeric(points[schema.rent], errors="coerce")
    points = points.dropna(subset=["_lat", "_lon", "_rent"])

    labels = points[schema.name].astype(str) if schema.name else points.index.astype(str)

    fig = go.Figure(
        go.Scattergeo(
            lat=points["_lat"],
            lon=points["_lon"],
            text=labels,
            mode="markers",
            marker=dict(
                size=10,
                color=points["_rent"],
                colorscale="Blues",
                colorbar=dict(title="Rent", tickprefix="$"),
                line=dict(width=0.5, color="white"),
            ),
            hovertemplate="%{text}<br>Rent $%{marker.color:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Comparable properties",
        geo=dict(scope="usa"),
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", color=INK),
        margin=dict(l=10, r=10, t=60, b=10),
        height=460,
    )
    return fig

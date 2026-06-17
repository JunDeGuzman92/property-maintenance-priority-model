"""Plotly chart builders for the rent-comparison dashboard.

Each function takes a tidy DataFrame (plus the column names that hold the data)
and returns a ready-to-render Plotly figure. Keeping the chart code here means
the Streamlit app stays focused on layout and the charts stay easy to reuse or
restyle in one place.

The styling aims for a calm, executive-ready look: muted colours, clear
dollar/percent labels, and no chart-junk.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# A small, consistent palette so the whole dashboard reads as one tool.
INK = "#1f2937"          # near-black for text
SUBJECT_BLUE = "#2563eb"  # the property under review
MARKET_GREY = "#94a3b8"   # the market reference
BAND_FILL = "#e0e7ff"     # recommended-band shading

# Plain-English colours for market positioning.
POSITION_COLORS = {
    "Below market": "#f59e0b",   # amber - room to raise
    "Near market": "#16a34a",    # green - on target
    "Above market": "#dc2626",   # red - re-lease risk
    "Unknown": "#9ca3af",
}

# Shared layout applied to every figure for a consistent feel.
_BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=14, color=INK),
    margin=dict(l=60, r=30, t=60, b=40),
    title_font=dict(size=18),
    hoverlabel=dict(font_size=13),
)


def _apply_base(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(**_BASE_LAYOUT, height=height)
    return fig


def subject_vs_market(subject_name: str, subject_rent: float, market_median: float) -> go.Figure:
    """Side-by-side bars: the subject property's rent against the market median."""
    fig = go.Figure()
    fig.add_bar(
        x=[subject_name, "Market median"],
        y=[subject_rent, market_median],
        marker_color=[SUBJECT_BLUE, MARKET_GREY],
        text=[f"${subject_rent:,.0f}", f"${market_median:,.0f}"],
        textposition="outside",
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    )
    fig.update_yaxes(title_text="Monthly rent (USD)", tickprefix="$", separatethousands=True)
    fig.update_layout(title="Subject rent vs. market median")
    return _apply_base(fig)


def recommended_band(band: dict, current_rent: float | None = None) -> go.Figure:
    """Horizontal view of the conservative / market / premium pricing band.

    The current rent (if known) is dropped on as a marker so the reader can see
    instantly where today's price sits inside the recommended range.
    """
    conservative = band["conservative"]
    premium = band["premium"]
    market = band["market"]

    fig = go.Figure()
    # The shaded band from conservative to premium.
    fig.add_shape(
        type="rect",
        x0=conservative, x1=premium, y0=0.3, y1=0.7,
        fillcolor=BAND_FILL, line_width=0, layer="below",
    )
    # Markers for the three reference points.
    fig.add_trace(
        go.Scatter(
            x=[conservative, market, premium],
            y=[0.5, 0.5, 0.5],
            mode="markers+text",
            marker=dict(size=14, color=[MARKET_GREY, SUBJECT_BLUE, MARKET_GREY], symbol="line-ns-open"),
            text=[
                f"Conservative<br>${conservative:,.0f}",
                f"Market<br>${market:,.0f}",
                f"Premium<br>${premium:,.0f}",
            ],
            textposition="top center",
            hoverinfo="skip",
        )
    )
    if current_rent is not None and pd.notna(current_rent):
        fig.add_trace(
            go.Scatter(
                x=[current_rent], y=[0.5],
                mode="markers+text",
                marker=dict(size=16, color=INK, symbol="diamond"),
                text=[f"Current<br>${current_rent:,.0f}"],
                textposition="bottom center",
                hovertemplate="Current rent $%{x:,.0f}<extra></extra>",
            )
        )
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_xaxes(title_text="Monthly rent (USD)", tickprefix="$", separatethousands=True)
    fig.update_layout(title="Recommended rent band", showlegend=False)
    return _apply_base(fig, height=300)


def gap_position_summary(df: pd.DataFrame, position_col: str = "rent_position") -> go.Figure:
    """Count of markets sitting below / near / above the market median."""
    order = ["Below market", "Near market", "Above market"]
    counts = df[position_col].value_counts().reindex(order).fillna(0)
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[POSITION_COLORS[p] for p in counts.index],
            text=counts.values.astype(int),
            textposition="outside",
            hovertemplate="%{x}<br>%{y} markets<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Number of markets")
    fig.update_layout(title="Rent gap analysis: where markets sit vs. median")
    return _apply_base(fig)


def rent_by_unit_type(df: pd.DataFrame, unit_type_col: str, rent_col: str) -> go.Figure:
    """Median rent for each unit type, sorted from least to most expensive."""
    grouped = (
        df.groupby(unit_type_col)[rent_col]
        .median()
        .sort_values()
        .reset_index()
    )
    fig = go.Figure(
        go.Bar(
            x=grouped[rent_col],
            y=grouped[unit_type_col].astype(str),
            orientation="h",
            marker_color=SUBJECT_BLUE,
            text=[f"${v:,.0f}" for v in grouped[rent_col]],
            textposition="outside",
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Median rent (USD)", tickprefix="$", separatethousands=True)
    fig.update_layout(title="Median rent by unit type")
    return _apply_base(fig)


def rent_per_sqft(df: pd.DataFrame, label_col: str, rent_per_sqft_col: str, top_n: int = 15) -> go.Figure:
    """Rent per square foot, a like-for-like efficiency comparison."""
    top = df.dropna(subset=[rent_per_sqft_col]).nlargest(top_n, rent_per_sqft_col)
    fig = go.Figure(
        go.Bar(
            x=top[rent_per_sqft_col],
            y=top[label_col].astype(str),
            orientation="h",
            marker_color="#0ea5e9",
            text=[f"${v:,.2f}" for v in top[rent_per_sqft_col]],
            textposition="outside",
            hovertemplate="%{y}<br>$%{x:,.2f} / sq ft<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Rent per square foot (USD)", tickprefix="$")
    fig.update_layout(title="Rent per square foot")
    return _apply_base(fig)


def rent_distribution(df: pd.DataFrame, rent_col: str, market_median: float | None = None) -> go.Figure:
    """Spread of rents across the comparison set, with the median called out."""
    rents = pd.to_numeric(df[rent_col], errors="coerce").dropna()
    fig = go.Figure(
        go.Histogram(
            x=rents,
            nbinsx=30,
            marker_color=SUBJECT_BLUE,
            opacity=0.85,
            hovertemplate="$%{x:,.0f}<br>%{y} markets<extra></extra>",
        )
    )
    if market_median is not None and pd.notna(market_median):
        fig.add_vline(
            x=market_median,
            line_dash="dash",
            line_color=INK,
            annotation_text=f"Median ${market_median:,.0f}",
            annotation_position="top",
        )
    fig.update_xaxes(title_text="Monthly rent (USD)", tickprefix="$", separatethousands=True)
    fig.update_yaxes(title_text="Number of markets")
    fig.update_layout(title="Distribution of market rents")
    return _apply_base(fig)


def amenity_prevalence(df: pd.DataFrame, amenity_cols: list[str]) -> go.Figure:
    """Share of comparable properties that offer each amenity."""
    shares = []
    for col in amenity_cols:
        flag = pd.to_numeric(df[col], errors="coerce").fillna(0)
        shares.append((col.replace("_", " ").title(), float(flag.mean()) * 100.0))
    table = pd.DataFrame(shares, columns=["amenity", "share"]).sort_values("share")
    fig = go.Figure(
        go.Bar(
            x=table["share"],
            y=table["amenity"],
            orientation="h",
            marker_color="#10b981",
            text=[f"{v:.0f}%" for v in table["share"]],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.0f}% of comps<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Share of comparable properties", ticksuffix="%", range=[0, 105])
    fig.update_layout(title="Amenity coverage across comparables")
    return _apply_base(fig)

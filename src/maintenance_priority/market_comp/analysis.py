"""Shared analytics for the rent-comparison dashboard.

This module sits between the raw data and the presentation layer
(`visuals.py`, `tables.py`, `maps.py`). It does two jobs:

1. Figure out what each dataset actually contains (schema detection), so the
   dashboard can light up the right sections and explain what is missing.
2. Provide the transparent rent math the leasing team relies on - the gap to
   market, the below / near / above classification, and a recommended rent
   band built from real comparable rents.

Nothing here invents data. If a column is not present, the helpers say so
rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# Common amenity flags we recognise if a richer internal comp sheet is loaded.
# These are only used when the columns are genuinely present in the file.
KNOWN_AMENITY_FLAGS = [
    "pool",
    "gym",
    "fitness",
    "parking",
    "garage",
    "laundry",
    "in_unit_laundry",
    "washer_dryer",
    "dishwasher",
    "balcony",
    "patio",
    "air_conditioning",
    "ac",
    "central_air",
    "pet_friendly",
    "pets_allowed",
    "elevator",
    "doorman",
    "concierge",
    "rooftop",
    "ev_charging",
    "walk_in_closet",
    "hardwood",
    "stainless_steel",
]


@dataclass
class MarketSchema:
    """Maps the business concepts the dashboard cares about to real columns.

    Every attribute is either a column name found in the data or ``None`` when
    that information is not available. The dashboard reads this to decide which
    sections to show and which to replace with a "what's needed" note.
    """

    name: str | None = None
    rent: str | None = None
    rent_estimate: str | None = None
    unit_type: str | None = None
    sqft: str | None = None
    state: str | None = None
    city: str | None = None
    latitude: str | None = None
    longitude: str | None = None
    postal: str | None = None
    yoy_growth: str | None = None
    size_rank: str | None = None
    region_type: str | None = None
    pressure_score: str | None = None
    review_note: str | None = None
    amenity_flags: list[str] = field(default_factory=list)
    amenity_text: str | None = None

    @property
    def has_location(self) -> bool:
        """True when we have something real to put on a map."""
        return bool((self.latitude and self.longitude) or self.state)

    @property
    def has_amenities(self) -> bool:
        return bool(self.amenity_flags or self.amenity_text)


def _normalise(column: str) -> str:
    """Lower-case a column name and strip separators for fuzzy matching."""
    return "".join(ch for ch in column.lower() if ch.isalnum())


def _first_match(columns, candidates):
    """Return the first real column whose normalised name matches a candidate."""
    lookup = {_normalise(col): col for col in columns}
    for candidate in candidates:
        key = _normalise(candidate)
        if key in lookup:
            return lookup[key]
    return None


def resolve_schema(df: pd.DataFrame) -> MarketSchema:
    """Inspect a DataFrame and report which business fields are available.

    The matching is forgiving on purpose: a leasing export might call the rent
    column ``asking_rent`` while the public benchmark calls it
    ``observed_rent``. Both should resolve to the same concept.
    """
    columns = list(df.columns)

    schema = MarketSchema()
    schema.name = _first_match(
        columns,
        ["property_name", "property", "community", "community_name",
         "comp_name", "name", "RegionName", "market", "address"],
    )
    schema.rent = _first_match(
        columns,
        ["current_rent", "asking_rent", "in_place_rent", "effective_rent",
         "monthly_rent", "subject_current_rent", "observed_rent", "rent",
         "rent_amount"],
    )
    schema.rent_estimate = _first_match(
        columns, ["estimated_market_rent", "market_rent_estimate", "comp_rent"]
    )
    schema.unit_type = _first_match(
        columns,
        ["unit_type", "floorplan", "floor_plan", "unit_mix", "bedrooms",
         "beds", "bed_count", "layout", "plan_type", "subject_unit_type"],
    )
    schema.sqft = _first_match(
        columns,
        ["sqft", "square_feet", "square_footage", "sq_ft", "living_area_sqft",
         "floor_area_sqft", "area_sqft", "subject_sqft", "unit_sqft"],
    )
    schema.state = _first_match(columns, ["StateName", "state", "state_code", "province"])
    schema.city = _first_match(columns, ["city", "town", "municipality"])
    schema.latitude = _first_match(columns, ["latitude", "lat"])
    schema.longitude = _first_match(columns, ["longitude", "lon", "lng", "long"])
    schema.postal = _first_match(columns, ["zip", "zipcode", "zip_code", "postal_code", "postcode"])
    schema.yoy_growth = _first_match(columns, ["yoy_growth_pct", "yoy_growth", "rent_growth_pct", "annual_growth_pct"])
    schema.size_rank = _first_match(columns, ["SizeRank", "size_rank", "market_size_rank"])
    schema.region_type = _first_match(columns, ["RegionType", "region_type", "market_type"])
    schema.pressure_score = _first_match(columns, ["market_pressure_score", "pressure_score"])
    schema.review_note = _first_match(columns, ["review_note", "review_flag", "note"])

    # Amenities: prefer an explicit text column, otherwise collect any
    # recognised on/off amenity flags that happen to be present.
    schema.amenity_text = _first_match(columns, ["amenities", "amenity_list", "features"])
    normalised = {_normalise(col): col for col in columns}
    schema.amenity_flags = [
        normalised[_normalise(flag)] for flag in KNOWN_AMENITY_FLAGS if _normalise(flag) in normalised
    ]

    return schema


def market_benchmark(df: pd.DataFrame, rent_col: str) -> float:
    """Median rent across the comparison set - our 'market' reference point."""
    rents = pd.to_numeric(df[rent_col], errors="coerce").dropna()
    if rents.empty:
        return float("nan")
    return float(rents.median())


def add_rent_gap(df: pd.DataFrame, rent_col: str, benchmark: float, band_pct: float = 3.0) -> pd.DataFrame:
    """Add the gap to market and a below / near / above label for each row.

    ``band_pct`` is the tolerance (in percent) around the market median that we
    still treat as "priced at market". Leasing teams usually work with a small
    band rather than an exact number, so this is adjustable in the dashboard.
    """
    out = df.copy()
    rents = pd.to_numeric(out[rent_col], errors="coerce")
    out["rent_gap_pct"] = (rents - benchmark) / benchmark * 100.0
    out["rent_position"] = out["rent_gap_pct"].apply(lambda gap: classify_position(gap, band_pct))
    out["suggested_action"] = out["rent_position"].map(SUGGESTED_ACTIONS)
    return out


def classify_position(gap_pct: float, band_pct: float) -> str:
    """Translate a percentage gap into plain-English market positioning."""
    if pd.isna(gap_pct):
        return "Unknown"
    if gap_pct < -band_pct:
        return "Below market"
    if gap_pct > band_pct:
        return "Above market"
    return "Near market"


SUGGESTED_ACTIONS = {
    "Below market": "Review for increase",
    "Near market": "Hold at market",
    "Above market": "Monitor re-lease risk",
    "Unknown": "Needs review",
}


def recommend_rent_band(peer_rents) -> dict:
    """Build a conservative / market / premium band from comparable rents.

    The band is read straight off the distribution of real comparable rents -
    the 25th, 50th and 75th percentiles - so it is defensible in a pricing
    conversation and contains no made-up multipliers.
    """
    rents = pd.to_numeric(pd.Series(peer_rents), errors="coerce").dropna()
    if rents.empty:
        return {"conservative": np.nan, "market": np.nan, "premium": np.nan, "sample_size": 0}
    return {
        "conservative": float(rents.quantile(0.25)),
        "market": float(rents.quantile(0.50)),
        "premium": float(rents.quantile(0.75)),
        "sample_size": int(rents.size),
    }


def add_market_tier(df: pd.DataFrame, size_rank_col: str | None) -> pd.DataFrame:
    """Group markets into readable size tiers from their population size rank.

    Smaller SizeRank means a larger metro, so tier 1 is the biggest markets.
    Used for the optional 'similar size' peer group in the dashboard.
    """
    out = df.copy()
    if not size_rank_col or size_rank_col not in out.columns:
        out["market_tier"] = "All markets"
        return out

    rank = pd.to_numeric(out[size_rank_col], errors="coerce")
    bins = [-np.inf, 50, 150, 300, np.inf]
    labels = ["Top 50 metros", "Large metros", "Mid-size metros", "Smaller metros"]
    out["market_tier"] = pd.cut(rank, bins=bins, labels=labels)
    return out

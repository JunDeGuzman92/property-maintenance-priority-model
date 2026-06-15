import pandas as pd


COMPLAINT_TYPES = [
    "APPLIANCE",
    "DOOR/WINDOW",
    "ELECTRIC",
    "FLOORING/STAIRS",
    "GENERAL",
    "HEAT/HOT WATER",
    "PAINT/PLASTER",
    "PLUMBING",
    "UNSANITARY CONDITION",
    "WATER LEAK",
]

BOROUGHS = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
CHANNELS = ["MOBILE", "ONLINE", "PHONE", "UNKNOWN"]


def build_feature_matrix(df):
    """Convert public service-request rows into a numeric ML feature matrix."""
    features = pd.DataFrame(index=df.index)
    features["created_month"] = pd.to_numeric(df.get("created_month", 0), errors="coerce").fillna(0)
    features["is_winter"] = pd.to_numeric(df.get("is_winter", 0), errors="coerce").fillna(0)
    features["has_descriptor"] = pd.to_numeric(df.get("has_descriptor", 0), errors="coerce").fillna(0)
    location = df.get("location_type", pd.Series("", index=df.index)).astype(str).str.upper()
    features["is_residential_location"] = location.str.contains("RESIDENTIAL|APARTMENT|BUILDING", na=False).astype(int)

    complaint_type = df.get("complaint_type", pd.Series("", index=df.index)).astype(str).str.upper()
    for complaint in COMPLAINT_TYPES:
        name = complaint.lower().replace("/", "_").replace(" ", "_").replace("-", "_")
        features[f"complaint_{name}"] = complaint_type.eq(complaint).astype(int)

    borough = df.get("borough", pd.Series("", index=df.index)).astype(str).str.upper()
    for boro in BOROUGHS:
        name = boro.lower().replace(" ", "_")
        features[f"borough_{name}"] = borough.eq(boro).astype(int)

    channel = df.get("open_data_channel_type", pd.Series("UNKNOWN", index=df.index)).fillna("UNKNOWN").astype(str).str.upper()
    for value in CHANNELS:
        features[f"channel_{value.lower()}"] = channel.eq(value).astype(int)
    return features

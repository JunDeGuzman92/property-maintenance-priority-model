from datetime import datetime

import numpy as np
import pandas as pd


CLOSED_STATUSES = {"CLOSED", "COMPLETE", "COMPLETED", "RESOLVED", "CANCELLED", "CANCELED"}


def first_existing_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def add_open_flag(df, status_col=None):
    output = df.copy()
    if status_col and status_col in output.columns:
        status = output[status_col].fillna("").astype(str).str.upper().str.strip()
        output["_is_open"] = ~status.isin(CLOSED_STATUSES)
        return output, ""

    if "is_open" in output.columns:
        output["_is_open"] = output["is_open"].astype(bool)
        return output, ""

    output["_is_open"] = True
    return output, "No status or is_open column was found, so backlog charts count all rows."


def add_aging_bucket(df):
    output = df.copy()
    age_col = first_existing_column(output, ["age_days", "days_open", "work_order_age_days"])
    note = ""

    if age_col:
        age_days = pd.to_numeric(output[age_col], errors="coerce")
    elif "created_date" in output.columns:
        created = pd.to_datetime(output["created_date"], errors="coerce")
        age_days = (pd.Timestamp(datetime.now()) - created).dt.total_seconds() / 86400.0
        note = "Aging buckets are based on created_date."
    elif "closure_days" in output.columns:
        age_days = pd.to_numeric(output["closure_days"], errors="coerce")
        note = "Aging buckets use closure_days because created/open age fields are not available."
    else:
        output["age_days"] = np.nan
        output["aging_bucket"] = "Age unavailable"
        return output, "age_days", "No age, created_date, or closure_days column was found."

    output["age_days"] = age_days.clip(lower=0)
    bins = [-0.01, 3, 7, 14, np.inf]
    labels = ["0-3 days", "4-7 days", "8-14 days", "15+ days"]
    output["aging_bucket"] = pd.cut(output["age_days"], bins=bins, labels=labels).astype("object")
    output["aging_bucket"] = output["aging_bucket"].fillna("Age unavailable")
    return output, "age_days", note


def recommended_action(row, score_col="priority_score", threshold=0.5):
    score = row.get(score_col, np.nan)
    age_days = row.get("age_days", np.nan)
    is_open = bool(row.get("_is_open", True))

    if not is_open:
        return "No backlog action"
    if pd.notna(score) and score >= max(threshold, 0.75):
        return "Review today"
    if pd.notna(age_days) and age_days >= 15:
        return "Escalate aging backlog"
    if pd.notna(score) and score >= threshold:
        return "Supervisor review"
    if pd.notna(age_days) and age_days >= 8:
        return "Confirm owner and next step"
    return "Standard queue"


def build_triage_table(df, score_col="priority_score", threshold=0.5, top_n=50):
    if df.empty:
        return pd.DataFrame()

    table = df.copy()
    if score_col in table.columns:
        table = table.sort_values(score_col, ascending=False, na_position="last")
    elif "age_days" in table.columns:
        table = table.sort_values("age_days", ascending=False, na_position="last")

    table = table.head(top_n).copy()
    table["priority_rank"] = range(1, len(table) + 1)
    table["recommended_action"] = table.apply(
        recommended_action,
        axis=1,
        score_col=score_col,
        threshold=threshold,
    )

    candidate_columns = [
        "priority_rank",
        "unique_key",
        "property",
        "property_id",
        "property_name",
        "building",
        "building_id",
        "building_name",
        "unit",
        "unit_id",
        "unit_number",
        "borough",
        "incident_zip",
        "location_type",
        "complaint_type",
        "category",
        "issue_category",
        "descriptor",
        "status",
        "aging_bucket",
        "age_days",
        score_col,
        "review_label",
        "priority_followup",
        "recommended_action",
    ]
    display_columns = []
    for column in candidate_columns:
        if column in table.columns and column not in display_columns:
            display_columns.append(column)

    output = table[display_columns].copy()
    rename_map = {
        "priority_rank": "Rank",
        "unique_key": "Request ID",
        "property": "Property",
        "property_id": "Property ID",
        "property_name": "Property",
        "building": "Building",
        "building_id": "Building ID",
        "building_name": "Building",
        "unit": "Unit",
        "unit_id": "Unit ID",
        "unit_number": "Unit",
        "borough": "Borough",
        "incident_zip": "ZIP",
        "location_type": "Location Type",
        "complaint_type": "Issue Category",
        "category": "Issue Category",
        "issue_category": "Issue Category",
        "descriptor": "Issue Detail",
        "status": "Status",
        "aging_bucket": "Aging Bucket",
        "age_days": "Age Days",
        score_col: "Priority Score",
        "review_label": "Review Label",
        "priority_followup": "Training Target",
        "recommended_action": "Recommended Action",
    }
    return output.rename(columns=rename_map)


def style_executive_table(table):
    if table.empty:
        return table

    styler = table.style.format(
        {
            "Priority Score": "{:.1%}",
            "Age Days": "{:.1f}",
        },
        na_rep="",
    )
    if "Priority Score" in table.columns:
        styler = styler.background_gradient(subset=["Priority Score"], cmap="Blues")
    if "Rank" in table.columns:
        styler = styler.set_properties(subset=["Rank"], **{"font-weight": "600"})
    try:
        return styler.hide(axis="index")
    except AttributeError:
        return styler

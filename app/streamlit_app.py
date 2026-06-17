import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

from download_data import build_training_table
from explainability import contribution_summary, optional_shap_summary, row_contributions, score_dataframe
from maintenance_priority.inference import load_model
from tables import add_aging_bucket, add_open_flag, build_triage_table, first_existing_column, style_executive_table
from visuals import (
    plot_aging_buckets,
    plot_confusion_matrix,
    plot_feature_contributions,
    plot_group_count,
    plot_issue_breakdown,
    plot_metrics_summary,
    plot_precision_recall_curve,
    plot_priority_ranking,
    plot_roc_curve,
    plot_status_summary,
)


DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "nyc_311_hpd_priority_training.csv"
RAW_DATA_PATH = ROOT / "data" / "raw" / "nyc_311_hpd_housing_requests.csv"
MODEL_PATH = ROOT / "artifacts" / "maintenance_priority_model.json"

MODEL_SOURCE_COLUMNS = [
    "created_month",
    "is_winter",
    "has_descriptor",
    "complaint_type",
    "borough",
    "location_type",
    "open_data_channel_type",
]

PROPERTY_GROUP_COLUMNS = [
    "property",
    "property_id",
    "property_name",
    "building",
    "building_id",
    "building_name",
    "unit",
    "unit_id",
    "unit_number",
]

FALLBACK_GROUP_COLUMNS = ["borough", "incident_zip", "location_type"]
CATEGORY_COLUMNS = ["complaint_type", "category", "issue_category", "request_type", "problem_type"]
STATUS_COLUMNS = ["status", "work_order_status", "order_status"]


st.set_page_config(
    page_title="Maintenance Priority Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.4rem;
        max-width: 1440px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.65rem;
    }
    [data-testid="stMetricLabel"] {
        color: #475569;
    }
    div[data-testid="stAlert"] {
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def load_model_payload(path):
    if not Path(path).exists():
        return None
    return load_model(path)


def prepare_dataframe(df):
    notes = []
    raw_markers = {"created_date", "closed_date", "complaint_type"}
    processed_markers = {"created_month", "priority_followup"}
    if raw_markers.issubset(df.columns) and not processed_markers.issubset(df.columns):
        try:
            return build_training_table(df), ["Raw service-request data was prepared with the existing training transform."]
        except Exception as exc:
            notes.append(f"Raw data transform was skipped: {exc}")
    return df.copy(), notes


def filter_dataframe(df, status_col, category_col, group_col):
    filtered = df.copy()
    with st.sidebar:
        st.subheader("Filters")
        if status_col and status_col in filtered.columns:
            statuses = sorted(filtered[status_col].dropna().astype(str).unique())
            selected_statuses = st.multiselect("Status", statuses, default=statuses)
            if selected_statuses:
                filtered = filtered[filtered[status_col].astype(str).isin(selected_statuses)]

        if category_col and category_col in filtered.columns:
            categories = sorted(filtered[category_col].dropna().astype(str).unique())
            default_categories = categories[: min(len(categories), 12)]
            selected_categories = st.multiselect("Issue category", categories, default=default_categories)
            if selected_categories:
                filtered = filtered[filtered[category_col].astype(str).isin(selected_categories)]

        if group_col and group_col in filtered.columns:
            groups = sorted(filtered[group_col].dropna().astype(str).unique())
            if len(groups) <= 80:
                selected_groups = st.multiselect(group_col.replace("_", " ").title(), groups, default=groups)
                if selected_groups:
                    filtered = filtered[filtered[group_col].astype(str).isin(selected_groups)]
    return filtered


def metric_value(value):
    return "0" if pd.isna(value) else f"{int(value):,}"


def main():
    st.title("Maintenance Priority Dashboard")
    st.caption("Maintenance triage, backlog visibility, and model diagnostics for property operations review.")

    with st.sidebar:
        st.header("Dashboard Controls")
        uploaded_file = st.file_uploader("Use a CSV export", type=["csv"])
        threshold = st.slider("Urgent review threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
        top_n = st.slider("Rows in triage table", min_value=10, max_value=100, value=40, step=10)

    if uploaded_file is not None:
        raw_df = load_uploaded_csv(uploaded_file)
        data_label = uploaded_file.name
    elif DEFAULT_DATA_PATH.exists():
        raw_df = load_csv(DEFAULT_DATA_PATH)
        data_label = str(DEFAULT_DATA_PATH.relative_to(ROOT))
    elif RAW_DATA_PATH.exists():
        raw_df = load_csv(RAW_DATA_PATH)
        data_label = str(RAW_DATA_PATH.relative_to(ROOT))
    else:
        st.error("No CSV data was found. Run scripts/download_data.py or upload a CSV export.")
        return

    df, prep_notes = prepare_dataframe(raw_df)
    for note in prep_notes:
        st.info(note)

    model_payload = load_model_payload(MODEL_PATH)
    missing_source_columns = [column for column in MODEL_SOURCE_COLUMNS if column not in df.columns]
    score_col = "priority_score"
    model_warnings = []

    if model_payload:
        df, model_warnings = score_dataframe(df, model_payload, threshold=threshold)
        if missing_source_columns:
            st.warning(
                "Model scoring used default values for missing input columns: "
                + ", ".join(missing_source_columns)
            )
    else:
        score_col = first_existing_column(df, ["priority_score", "urgent_followup_score", "score"])
        st.warning("Model artifact was not found, so model-scored sections use any score column in the data.")

    for warning in model_warnings:
        st.warning(warning)

    status_col = first_existing_column(df, STATUS_COLUMNS)
    category_col = first_existing_column(df, CATEGORY_COLUMNS)
    property_group_col = first_existing_column(df, PROPERTY_GROUP_COLUMNS)
    fallback_group_col = first_existing_column(df, FALLBACK_GROUP_COLUMNS)
    group_col = property_group_col or fallback_group_col

    df, open_note = add_open_flag(df, status_col=status_col)
    df, age_col, age_note = add_aging_bucket(df)
    if open_note:
        st.warning(open_note)
    if age_note:
        st.info(age_note)
    if not property_group_col and fallback_group_col:
        st.info(
            "Property, building, and unit fields were not found; open-work-order grouping uses "
            f"{fallback_group_col}."
        )

    filtered = filter_dataframe(df, status_col, category_col, group_col)
    open_df = filtered[filtered["_is_open"]].copy() if "_is_open" in filtered.columns else filtered

    st.write(f"Data source: `{data_label}`")
    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    high_priority_count = int((filtered[score_col] >= threshold).sum()) if score_col in filtered.columns else 0
    median_age = filtered[age_col].median() if age_col in filtered.columns else pd.NA
    backlog_count = int(filtered["_is_open"].sum()) if "_is_open" in filtered.columns else len(filtered)

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Work orders", metric_value(len(filtered)))
    kpi_cols[1].metric("Open backlog", metric_value(backlog_count))
    kpi_cols[2].metric("Urgent review queue", metric_value(high_priority_count))
    kpi_cols[3].metric("Median age days", "N/A" if pd.isna(median_age) else f"{median_age:.1f}")

    triage_tab, operations_tab, diagnostics_tab = st.tabs(["Triage", "Operations", "Model Diagnostics"])

    with triage_tab:
        left, right = st.columns([1.15, 0.85])
        with left:
            st.plotly_chart(
                plot_priority_ranking(filtered, score_col=score_col, top_n=min(top_n, 40)),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(plot_aging_buckets(open_df), use_container_width=True)

        st.subheader("Executive Triage Table")
        triage_table = build_triage_table(filtered, score_col=score_col, threshold=threshold, top_n=top_n)
        st.dataframe(style_executive_table(triage_table), use_container_width=True, height=520)

    with operations_tab:
        left, right = st.columns(2)
        with left:
            group_title = f"Open Work Orders By {group_col.replace('_', ' ').title()}" if group_col else "Open Work Orders"
            st.plotly_chart(plot_group_count(open_df, group_col, group_title), use_container_width=True)
            st.plotly_chart(plot_status_summary(filtered, status_col), use_container_width=True)
        with right:
            st.plotly_chart(plot_issue_breakdown(filtered, category_col), use_container_width=True)
            st.plotly_chart(plot_aging_buckets(filtered), use_container_width=True)

        st.subheader("Recommended Action List")
        actions = build_triage_table(open_df, score_col=score_col, threshold=threshold, top_n=15)
        action_columns = [column for column in ["Rank", "Request ID", "Issue Category", "Status", "Aging Bucket", "Priority Score", "Recommended Action"] if column in actions.columns]
        st.dataframe(style_executive_table(actions[action_columns]), use_container_width=True, height=360)

    with diagnostics_tab:
        if not model_payload:
            st.warning("Model diagnostics require artifacts/maintenance_priority_model.json.")
            return

        metrics = model_payload.get("metrics", {})
        left, right = st.columns(2)
        with left:
            st.plotly_chart(plot_metrics_summary(metrics), use_container_width=True)
        with right:
            st.plotly_chart(plot_confusion_matrix(metrics), use_container_width=True)

        if "priority_followup" in filtered.columns and score_col in filtered.columns:
            target = pd.to_numeric(filtered["priority_followup"], errors="coerce")
            scores = pd.to_numeric(filtered[score_col], errors="coerce")
            valid = target.notna() & scores.notna()
            roc_col, pr_col = st.columns(2)
            with roc_col:
                st.plotly_chart(plot_roc_curve(target[valid], scores[valid]), use_container_width=True)
            with pr_col:
                st.plotly_chart(plot_precision_recall_curve(target[valid], scores[valid]), use_container_width=True)
        else:
            st.info("ROC and precision-recall diagnostics need priority_followup and priority_score columns.")

        st.subheader("Model Explainability")
        contrib_df, contrib_warnings = contribution_summary(filtered, model_payload)
        for warning in contrib_warnings:
            st.warning(warning)

        left, right = st.columns(2)
        with left:
            st.plotly_chart(plot_feature_contributions(contrib_df), use_container_width=True)
        with right:
            top_row = filtered.sort_values(score_col, ascending=False).iloc[0] if score_col in filtered.columns else filtered.iloc[0]
            row_df, row_warnings = row_contributions(top_row, model_payload)
            for warning in row_warnings:
                st.warning(warning)
            st.plotly_chart(
                plot_feature_contributions(
                    row_df,
                    value_col="abs_contribution",
                    title="Top Drivers For Highest-Priority Row",
                ),
                use_container_width=True,
            )

        run_shap = st.checkbox("Run SHAP sample explanation", value=False)
        if run_shap:
            with st.spinner("Running SHAP on a small sample..."):
                shap_df, shap_message = optional_shap_summary(filtered, model_payload)
            if shap_message:
                st.warning(shap_message)
            else:
                st.plotly_chart(
                    plot_feature_contributions(
                        shap_df.head(15),
                        value_col="mean_abs_shap",
                        title="SHAP Sample Summary",
                    ),
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()

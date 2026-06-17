import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


MUTED_SEQUENCE = ["#2563eb", "#0f766e", "#7c3aed", "#d97706", "#475569", "#be123c"]


def _clean_layout(fig, title=None, height=380):
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=height,
        margin={"l": 16, "r": 16, "t": 54 if title else 24, "b": 36},
        font={"family": "Arial, sans-serif", "size": 13, "color": "#172033"},
        legend_title_text="",
        hoverlabel={"bgcolor": "white", "font_size": 12},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def empty_figure(title, message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": "#64748b"},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _clean_layout(fig, title=title, height=300)


def _row_label(row, label_columns):
    parts = []
    for column in label_columns:
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            parts.append(str(row[column]))
    label = " | ".join(parts) if parts else f"Row {row.name}"
    return label[:90] + "..." if len(label) > 90 else label


def plot_priority_ranking(df, score_col="priority_score", label_columns=None, top_n=25):
    title = "Priority Score Ranking"
    if score_col not in df.columns or df.empty:
        return empty_figure(title, "Priority scores are not available.")

    label_columns = label_columns or ["unique_key", "complaint_type", "borough", "incident_zip"]
    plot_df = df.sort_values(score_col, ascending=False).head(top_n).copy()
    plot_df["display_label"] = plot_df.apply(_row_label, axis=1, label_columns=label_columns)
    plot_df = plot_df.sort_values(score_col, ascending=True)

    fig = px.bar(
        plot_df,
        x=score_col,
        y="display_label",
        orientation="h",
        color=score_col,
        color_continuous_scale=["#dbeafe", "#1d4ed8"],
        range_x=[0, 1],
        labels={score_col: "Priority score", "display_label": ""},
    )
    fig.update_traces(hovertemplate="%{y}<br>Priority score: %{x:.1%}<extra></extra>")
    fig.update_layout(coloraxis_showscale=False)
    return _clean_layout(fig, title=title, height=max(360, min(720, 90 + len(plot_df) * 24)))


def plot_group_count(df, group_col, title, top_n=15):
    if not group_col or group_col not in df.columns or df.empty:
        return empty_figure(title, "No usable grouping column is available.")

    counts = (
        df[group_col]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(top_n)
        .sort_values(ascending=True)
        .reset_index()
    )
    counts.columns = [group_col, "count"]
    fig = px.bar(
        counts,
        x="count",
        y=group_col,
        orientation="h",
        color_discrete_sequence=["#0f766e"],
        labels={"count": "Open work orders", group_col: ""},
    )
    fig.update_traces(hovertemplate="%{y}<br>Open work orders: %{x}<extra></extra>")
    return _clean_layout(fig, title=title, height=max(330, min(620, 120 + len(counts) * 28)))


def plot_aging_buckets(df, bucket_col="aging_bucket"):
    title = "Aging Buckets"
    if bucket_col not in df.columns or df.empty:
        return empty_figure(title, "Aging buckets are not available.")

    order = ["0-3 days", "4-7 days", "8-14 days", "15+ days", "Age unavailable"]
    counts = df[bucket_col].value_counts().reindex(order).dropna().reset_index()
    counts.columns = [bucket_col, "count"]
    fig = px.bar(
        counts,
        x=bucket_col,
        y="count",
        color=bucket_col,
        color_discrete_sequence=MUTED_SEQUENCE,
        labels={bucket_col: "Bucket", "count": "Work orders"},
    )
    fig.update_traces(hovertemplate="%{x}<br>Work orders: %{y}<extra></extra>")
    fig.update_layout(showlegend=False)
    return _clean_layout(fig, title=title, height=340)


def plot_issue_breakdown(df, category_col, top_n=12):
    title = "Issue Category Breakdown"
    if not category_col or category_col not in df.columns or df.empty:
        return empty_figure(title, "No issue category column is available.")

    counts = (
        df[category_col]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(top_n)
        .sort_values(ascending=True)
        .reset_index()
    )
    counts.columns = [category_col, "count"]
    fig = px.bar(
        counts,
        x="count",
        y=category_col,
        orientation="h",
        color_discrete_sequence=["#2563eb"],
        labels={"count": "Work orders", category_col: ""},
    )
    fig.update_traces(hovertemplate="%{y}<br>Work orders: %{x}<extra></extra>")
    return _clean_layout(fig, title=title, height=max(340, min(620, 130 + len(counts) * 30)))


def plot_status_summary(df, status_col):
    title = "Status And Backlog Summary"
    if not status_col or status_col not in df.columns or df.empty:
        return empty_figure(title, "No status column is available.")

    counts = df[status_col].fillna("Unknown").astype(str).value_counts().reset_index()
    counts.columns = [status_col, "count"]
    fig = px.bar(
        counts,
        x=status_col,
        y="count",
        color=status_col,
        color_discrete_sequence=MUTED_SEQUENCE,
        labels={status_col: "Status", "count": "Work orders"},
    )
    fig.update_traces(hovertemplate="%{x}<br>Work orders: %{y}<extra></extra>")
    fig.update_layout(showlegend=False)
    return _clean_layout(fig, title=title, height=340)


def plot_metrics_summary(metrics):
    title = "Classification Metrics"
    keys = ["accuracy", "precision", "recall", "f1"]
    rows = [{"metric": key.title(), "value": metrics.get(key)} for key in keys if metrics.get(key) is not None]
    if not rows:
        return empty_figure(title, "Classification metrics are not available.")

    metrics_df = pd.DataFrame(rows)
    fig = px.bar(
        metrics_df,
        x="metric",
        y="value",
        color="metric",
        color_discrete_sequence=MUTED_SEQUENCE,
        range_y=[0, 1],
        labels={"metric": "", "value": "Score"},
    )
    fig.update_traces(hovertemplate="%{x}: %{y:.1%}<extra></extra>")
    fig.update_layout(showlegend=False)
    return _clean_layout(fig, title=title, height=330)


def plot_confusion_matrix(metrics):
    title = "Confusion Matrix"
    needed = ["tn", "fp", "fn", "tp"]
    if not all(key in metrics for key in needed):
        return empty_figure(title, "Confusion matrix counts are not available.")

    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    text = [[f"{labels[i][j]}<br>{matrix[i, j]:,}" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted Standard", "Predicted Urgent"],
            y=["Actual Standard", "Actual Urgent"],
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    return _clean_layout(fig, title=title, height=330)


def plot_roc_curve(y_true, y_score):
    title = "ROC Curve"
    try:
        from sklearn.metrics import auc, roc_curve
    except ImportError:
        return empty_figure(title, "Install scikit-learn to show ROC diagnostics.")

    if len(pd.Series(y_true).dropna().unique()) < 2:
        return empty_figure(title, "ROC requires both positive and negative target rows.")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC {roc_auc:.2f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line={"dash": "dash"}))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1])
    return _clean_layout(fig, title=title, height=330)


def plot_precision_recall_curve(y_true, y_score):
    title = "Precision-Recall Curve"
    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve
    except ImportError:
        return empty_figure(title, "Install scikit-learn to show precision-recall diagnostics.")

    if len(pd.Series(y_true).dropna().unique()) < 2:
        return empty_figure(title, "Precision-recall requires both positive and negative target rows.")

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"Average precision {avg_precision:.2f}",
        )
    )
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1])
    return _clean_layout(fig, title=title, height=330)


def plot_feature_contributions(contrib_df, value_col="mean_abs_contribution", title="Model Driver Summary"):
    if contrib_df.empty or value_col not in contrib_df.columns:
        return empty_figure(title, "Model contribution data is not available.")

    label_col = "business_label" if "business_label" in contrib_df.columns else "feature"
    plot_df = contrib_df.sort_values(value_col, ascending=True)
    fig = px.bar(
        plot_df,
        x=value_col,
        y=label_col,
        orientation="h",
        color_discrete_sequence=["#475569"],
        labels={value_col: "Average absolute impact", label_col: ""},
    )
    fig.update_traces(hovertemplate="%{y}<br>Impact: %{x:.4f}<extra></extra>")
    return _clean_layout(fig, title=title, height=max(340, min(650, 130 + len(plot_df) * 28)))

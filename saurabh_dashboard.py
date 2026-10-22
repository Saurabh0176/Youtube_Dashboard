# -*- coding: utf-8 -*-
"""
YouTube Analytics Dashboard
Run with: python -m streamlit run saurabh_dashboard.py
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
st.set_page_config(page_title="YouTube Analytics Dashboard", layout="wide")

DATA_DIR = Path(__file__).resolve().parent

AGG_FILE = DATA_DIR / "Aggregated_Metrics_By_Video.csv"
COUNTRY_FILE = DATA_DIR / "Aggregated_Metrics_By_Country_And_Subscriber_Status.csv"
TIME_FILE = DATA_DIR / "Video_Performance_Over_Time.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def style_negative(value, props=""):
    """Apply styling to negative numeric values."""
    try:
        return props if pd.notna(value) and value < 0 else None
    except (TypeError, ValueError):
        return None


def style_positive(value, props=""):
    """Apply styling to positive numeric values."""
    try:
        return props if pd.notna(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


def audience_simple(country):
    """Group countries into USA, India, and Other."""
    if country == "US":
        return "USA"
    if country == "IN":
        return "India"
    return "Other"


def parse_duration_seconds(value):
    """Convert HH:MM:SS duration strings to seconds."""
    if pd.isna(value):
        return np.nan
    try:
        parts = str(value).strip().split(":")
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def require_file(path):
    """Raise a clear error if a required CSV is missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find '{path.name}'. Put it in this folder:\n{path.parent}"
        )


# ---------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the three source CSVs and prepare the dashboard data."""
    for path in (AGG_FILE, COUNTRY_FILE, TIME_FILE):
        require_file(path)

    # Video-level aggregated metrics
    df_agg = pd.read_csv(AGG_FILE)
    # The original dataset has an extra first row that is not data.
    if not df_agg.empty:
        df_agg = df_agg.iloc[1:, :].copy()

    expected_columns = [
        "Video",
        "Video title",
        "Video publish time",
        "Comments added",
        "Shares",
        "Dislikes",
        "Likes",
        "Subscribers lost",
        "Subscribers gained",
        "RPM(USD)",
        "CPM(USD)",
        "Average % viewed",
        "Average view duration",
        "Views",
        "Watch time (hours)",
        "Subscribers",
        "Your estimated revenue (USD)",
        "Impressions",
        "Impressions ctr(%)",
    ]

    if len(df_agg.columns) == len(expected_columns):
        df_agg.columns = expected_columns

    if "Video publish time" not in df_agg.columns:
        raise ValueError(
            "The video metrics CSV does not contain the expected "
            "'Video publish time' column. Check the CSV headers."
        )

    df_agg["Video publish time"] = pd.to_datetime(
        df_agg["Video publish time"],
        format="mixed",
        errors="coerce",
    )

    # Convert metric columns to numeric where possible.
    numeric_candidates = [
        "Comments added", "Shares", "Dislikes", "Likes",
        "Subscribers lost", "Subscribers gained", "RPM(USD)",
        "CPM(USD)", "Average % viewed", "Views", "Watch time (hours)",
        "Subscribers", "Your estimated revenue (USD)", "Impressions",
        "Impressions ctr(%)",
    ]
    for col in numeric_candidates:
        if col in df_agg.columns:
            df_agg[col] = pd.to_numeric(df_agg[col], errors="coerce")

    if "Average view duration" in df_agg.columns:
        df_agg["Avg_duration_sec"] = df_agg["Average view duration"].apply(
            parse_duration_seconds
        )

    # Avoid division-by-zero errors.
    if {"Comments added", "Shares", "Dislikes", "Likes", "Views"}.issubset(df_agg.columns):
        df_agg["Engagement_ratio"] = (
            df_agg[["Comments added", "Shares", "Dislikes", "Likes"]]
            .sum(axis=1, min_count=1)
            .div(df_agg["Views"].replace(0, np.nan))
        )

    if {"Views", "Subscribers gained"}.issubset(df_agg.columns):
        df_agg["Views / sub gained"] = df_agg["Views"].div(
            df_agg["Subscribers gained"].replace(0, np.nan)
        )

    df_agg = df_agg.dropna(subset=["Video publish time"]).sort_values(
        "Video publish time", ascending=False
    ).reset_index(drop=True)

    # Country and subscriber-status data
    df_agg_sub = pd.read_csv(COUNTRY_FILE)

    # Daily video performance data
    df_time = pd.read_csv(TIME_FILE)
    if "Date" in df_time.columns:
        df_time["Date"] = pd.to_datetime(
            df_time["Date"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )

    if "Views" in df_time.columns:
        df_time["Views"] = pd.to_numeric(df_time["Views"], errors="coerce")

    return df_agg, df_agg_sub, df_time


# ---------------------------------------------------------------------
# Prepare data
# ---------------------------------------------------------------------
try:
    df_agg, df_agg_sub, df_time = load_data()
except Exception as exc:
    st.error(f"Unable to load dashboard data: {exc}")
    st.stop()

if df_agg.empty:
    st.error("No valid video rows were found in Aggregated_Metrics_By_Video.csv.")
    st.stop()

# Numeric columns used for relative-to-median comparison.
metric_columns = [
    col for col in [
        "Views", "Likes", "Subscribers", "Shares", "Comments added",
        "RPM(USD)", "Average % viewed", "Avg_duration_sec",
        "Engagement_ratio", "Views / sub gained",
    ]
    if col in df_agg.columns
]

# Create relative differences from the median of the last 12 months.
df_agg_diff = df_agg.copy()
latest_publish_date = df_agg["Video publish time"].max()
metric_date_12mo = latest_publish_date - pd.DateOffset(months=12)
recent_12mo = df_agg.loc[
    df_agg["Video publish time"] >= metric_date_12mo, metric_columns
]
median_agg = recent_12mo.median(numeric_only=True).replace(0, np.nan)

for col in metric_columns:
    if col in median_agg.index and pd.notna(median_agg[col]):
        df_agg_diff[col] = (
            pd.to_numeric(df_agg_diff[col], errors="coerce") - median_agg[col]
        ) / median_agg[col]

# Merge daily metrics with each video's publish date.
df_time_diff = pd.DataFrame()
if {"External Video ID", "Date", "Views"}.issubset(df_time.columns) and "Video" in df_agg.columns:
    publish_dates = df_agg[["Video", "Video publish time"]].drop_duplicates("Video")
    df_time_diff = df_time.merge(
        publish_dates,
        left_on="External Video ID",
        right_on="Video",
        how="left",
    )
    df_time_diff["days_published"] = (
        df_time_diff["Date"] - df_time_diff["Video publish time"]
    ).dt.days

    date_12mo = latest_publish_date - pd.DateOffset(months=12)
    df_time_diff_yr = df_time_diff[
        df_time_diff["Video publish time"] >= date_12mo
    ].copy()
else:
    df_time_diff_yr = pd.DataFrame()

# Typical cumulative views during the first 30 days after publishing.
if not df_time_diff_yr.empty:
    daily_views = df_time_diff_yr[
        df_time_diff_yr["days_published"].between(0, 30)
    ].copy()

    views_days = (
        daily_views.groupby("days_published")["Views"]
        .agg(
            mean_views="mean",
            median_views="median",
            pct80_views=lambda s: s.quantile(0.80),
            pct20_views=lambda s: s.quantile(0.20),
        )
        .reset_index()
        .sort_values("days_published")
    )

    views_cumulative = views_days[
        ["days_published", "median_views", "pct80_views", "pct20_views"]
    ].copy()
    views_cumulative[
        ["median_views", "pct80_views", "pct20_views"]
    ] = views_cumulative[
        ["median_views", "pct80_views", "pct20_views"]
    ].cumsum()
else:
    views_cumulative = pd.DataFrame(
        columns=["days_published", "median_views", "pct80_views", "pct20_views"]
    )


# ---------------------------------------------------------------------
# Streamlit dashboard
# ---------------------------------------------------------------------
st.title("YouTube Analytics Dashboard")

analysis_type = st.sidebar.selectbox(
    "Choose analysis",
    ("Aggregate Metrics", "Individual Video Analysis"),
)

if analysis_type == "Aggregate Metrics":
    st.subheader("YouTube Aggregated Metrics")

    # Select columns explicitly so date columns are never included in medians.
    df_agg_metrics = df_agg[["Video publish time"] + metric_columns].copy()
    metric_date_6mo = latest_publish_date - pd.DateOffset(months=6)
    metric_date_12mo = latest_publish_date - pd.DateOffset(months=12)

    recent_6mo = df_agg_metrics[
        df_agg_metrics["Video publish time"] >= metric_date_6mo
    ]
    recent_12mo_metrics = df_agg_metrics[
        df_agg_metrics["Video publish time"] >= metric_date_12mo
    ]

    medians_6mo = recent_6mo[metric_columns].median(numeric_only=True)
    medians_12mo = recent_12mo_metrics[metric_columns].median(numeric_only=True)

    # Show five metrics per row, with a safe percentage delta.
    display_metrics = [col for col in metric_columns if col in medians_6mo.index]
    for start in range(0, len(display_metrics), 5):
        metric_row = st.columns(5)
        for container, metric_name in zip(metric_row, display_metrics[start:start + 5]):
            new_value = medians_6mo.get(metric_name, np.nan)
            old_value = medians_12mo.get(metric_name, np.nan)

            if pd.notna(old_value) and old_value != 0 and pd.notna(new_value):
                delta = (new_value - old_value) / abs(old_value)
                delta_text = f"{delta:.2%}"
            else:
                delta_text = None

            with container:
                st.metric(
                    label=metric_name,
                    value=f"{new_value:,.1f}" if pd.notna(new_value) else "N/A",
                    delta=delta_text,
                )

    st.subheader("Video metrics relative to 12-month median")

    df_agg_diff["Publish_date"] = df_agg_diff["Video publish time"].dt.date
    table_columns = [
        col for col in [
            "Video title", "Publish_date", "Views", "Likes", "Subscribers",
            "Shares", "Comments added", "RPM(USD)", "Average % viewed",
            "Avg_duration_sec", "Engagement_ratio", "Views / sub gained",
        ]
        if col in df_agg_diff.columns
    ]
    df_table = df_agg_diff[table_columns].copy()

    percent_formats = {
        col: "{:.1%}".format
        for col in metric_columns
        if col in df_table.columns
    }

    styled_table = (
        df_table.style
        .map(style_negative, props="color: red;")
        .map(style_positive, props="color: green;")
        .format(percent_formats, na_rep="—")
    )
    st.dataframe(styled_table, use_container_width=True, hide_index=True)


elif analysis_type == "Individual Video Analysis":
    st.subheader("Individual Video Performance")

    if "Video title" not in df_agg.columns:
        st.error("The video metrics CSV is missing the 'Video title' column.")
        st.stop()

    video_titles = df_agg["Video title"].dropna().astype(str).unique().tolist()
    if not video_titles:
        st.warning("No video titles were found.")
        st.stop()

    selected_video = st.selectbox("Choose a video", video_titles)
    selected_rows = df_agg[df_agg["Video title"].astype(str) == selected_video]

    # Subscriber status and country chart.
    required_country_cols = {"Video Title", "Country Code", "Is Subscribed", "Views"}
    if required_country_cols.issubset(df_agg_sub.columns):
        country_rows = df_agg_sub[
            df_agg_sub["Video Title"].astype(str) == selected_video
        ].copy()

        if not country_rows.empty:
            country_rows["Country"] = country_rows["Country Code"].apply(audience_simple)
            country_rows["Views"] = pd.to_numeric(country_rows["Views"], errors="coerce")
            country_rows = country_rows.sort_values("Is Subscribed")

            fig = px.bar(
                country_rows,
                x="Views",
                y="Is Subscribed",
                color="Country",
                orientation="h",
                barmode="group",
                title="Views by subscriber status and country group",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No country/subscriber data was found for this video.")
    else:
        st.info(
            "The country CSV does not contain all expected columns "
            "('Video Title', 'Country Code', 'Is Subscribed', 'Views')."
        )

    # Compare this video's cumulative views with the typical first-30-day curve.
    if not df_time_diff.empty and "Video Title" in df_time_diff.columns:
        video_time = df_time_diff[
            df_time_diff["Video Title"].astype(str) == selected_video
        ].copy()
    elif not df_time_diff.empty and "External Video ID" in df_time_diff.columns:
        video_ids = selected_rows["Video"].astype(str).unique()
        video_time = df_time_diff[
            df_time_diff["External Video ID"].astype(str).isin(video_ids)
        ].copy()
    else:
        video_time = pd.DataFrame()

    if not video_time.empty and "days_published" in video_time.columns:
        first_30 = video_time[
            video_time["days_published"].between(0, 30)
        ].sort_values("days_published").copy()
        first_30["Views"] = pd.to_numeric(first_30["Views"], errors="coerce")

        fig2 = go.Figure()

        if not views_cumulative.empty:
            fig2.add_trace(go.Scatter(
                x=views_cumulative["days_published"],
                y=views_cumulative["pct20_views"],
                mode="lines",
                name="20th percentile",
                line=dict(color="purple", dash="dash"),
            ))
            fig2.add_trace(go.Scatter(
                x=views_cumulative["days_published"],
                y=views_cumulative["median_views"],
                mode="lines",
                name="50th percentile (median)",
                line=dict(color="black", dash="dash"),
            ))
            fig2.add_trace(go.Scatter(
                x=views_cumulative["days_published"],
                y=views_cumulative["pct80_views"],
                mode="lines",
                name="80th percentile",
                line=dict(color="royalblue", dash="dash"),
            ))

        if not first_30.empty:
            fig2.add_trace(go.Scatter(
                x=first_30["days_published"],
                y=first_30["Views"].cumsum(),
                mode="lines+markers",
                name="Selected video",
                line=dict(color="firebrick", width=4),
            ))

        fig2.update_layout(
            title="Cumulative views: first 30 days after publication",
            xaxis_title="Days since published",
            yaxis_title="Cumulative views",
            legend_title="Series",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Daily performance data could not be matched to this video.")
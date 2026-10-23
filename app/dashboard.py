"""
dashboard.py
------------
Streamlit dashboard for the Delhi load forecasting project.

Shows the model's day-ahead load forecast against the real load, lets the user
pick a peak threshold, and lists the days that would trigger a demand-response
alert.

Run it from the project folder with:
    streamlit run app/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demand_response import SUGGESTED_ACTION, alert_quality, find_alerts

APP_DIR = Path(__file__).resolve().parent
REPO_URL = "https://github.com/officialanas03/delhi-load-forecast-dashboard"

st.set_page_config(page_title="Delhi Load Forecast", layout="wide")


@st.cache_data
def load_files():
    data = pd.read_csv(APP_DIR / "forecast_data.csv", parse_dates=["timestamp"])
    meta = json.loads((APP_DIR / "metadata.json").read_text())
    return data, meta


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    return float((actual - predicted).abs().div(actual).mean() * 100)


try:
    data, meta = load_files()
except FileNotFoundError:
    st.error(
        "The data files for the dashboard were not found. "
        "Run `python3 src/make_dashboard_data.py` first."
    )
    st.stop()

first_day = data["timestamp"].min().date()
last_day = data["timestamp"].max().date()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")

# Open on the 30 days around the highest load, so the first view shows peak alerts
peak_time = data.loc[data["actual"].idxmax(), "timestamp"]
default_start = max(first_day, (peak_time - pd.Timedelta(days=15)).date())
default_end = min(last_day, default_start + pd.Timedelta(days=30))
date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start, default_end),
    min_value=first_day,
    max_value=last_day,
)
if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Pick both a start date and an end date in the sidebar.")
    st.stop()
start_day, end_day = date_range

low = int(data["forecast"].min() // 100 * 100)
high = int(-(-data["forecast"].max() // 100) * 100)
threshold = st.sidebar.slider(
    "Peak threshold (MW)",
    min_value=low,
    max_value=high,
    value=min(max(int(meta["default_threshold"]), low), high),
    step=100,
    help="Hours forecast at or above this load count as peak hours.",
)
show_last_week = st.sidebar.checkbox("Show 'same hour last week' for comparison", value=True)

view = data[(data["timestamp"].dt.date >= start_day) & (data["timestamp"].dt.date <= end_day)]
if view.empty:
    st.warning("No data in this date range.")
    st.stop()

# ---------------------------------------------------------------- header
st.title("Delhi Load Forecast & Peak Alerts")
st.caption(
    f"Day-ahead hourly forecasts for {meta['start']} to {meta['end']}, made by a "
    f"{meta['model_name']} model trained only on data before {meta['train_end']}."
)

scored = view.dropna(subset=["actual", "forecast", "last_week"])
alerts = find_alerts(view, threshold)
quality = alert_quality(view, threshold)

c1, c2, c3, c4 = st.columns(4)
if len(scored):
    c1.metric("Average error of the model", f"{mape(scored['actual'], scored['forecast']):.1f}%")
    c2.metric("Same hour last week", f"{mape(scored['actual'], scored['last_week']):.1f}%")
else:
    c1.metric("Average error of the model", "n/a")
    c2.metric("Same hour last week", "n/a")
c3.metric("Days with a peak alert", len(alerts))
c4.metric(
    "Real peak days caught",
    f"{quality['caught']} of {quality['real']}" if quality["real"] else "no real peaks",
)

tab_forecast, tab_alerts, tab_about = st.tabs(["Forecast", "Peak alerts", "About"])

# ---------------------------------------------------------------- forecast tab
with tab_forecast:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["actual"], name="Actual load",
                             line=dict(color="black", width=1.8)))
    fig.add_trace(go.Scatter(x=view["timestamp"], y=view["forecast"], name="Forecast",
                             line=dict(color="#2ca02c", width=1.6)))
    if show_last_week:
        fig.add_trace(go.Scatter(x=view["timestamp"], y=view["last_week"],
                                 name="Same hour last week",
                                 line=dict(color="#ff7f0e", width=1, dash="dot"), opacity=0.8))
    peaks = view[view["forecast"] >= threshold]
    fig.add_trace(go.Scatter(x=peaks["timestamp"], y=peaks["forecast"], mode="markers",
                             name="Forecast peak hour",
                             marker=dict(color="red", size=6)))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold {threshold:,} MW", annotation_position="top left")
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Load (MW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig)
    st.caption(
        "Gaps in the black line are hours where the load data is missing. "
        "Forecast values use the real temperature of each day, as if the weather forecast were perfect."
    )

# ---------------------------------------------------------------- alerts tab
with tab_alerts:
    st.subheader("Peak alerts")
    if alerts.empty:
        st.success(f"No hours are forecast at or above {threshold:,} MW in this period.")
    else:
        st.write(
            f"{len(alerts)} day(s) in this period have forecast hours at or above "
            f"{threshold:,} MW."
        )
        st.dataframe(alerts, hide_index=True)

    st.subheader("How good were the alerts?")
    if quality["days"] == 0:
        st.write("There is not enough data in this period to check the alerts.")
    else:
        text = (
            f"Out of {quality['days']} days with almost complete load data, the forecast flagged {quality['flagged']} "
            f"and {quality['real']} really went above {threshold:,} MW. "
            f"The alerts caught {quality['caught']} of the real peak days, "
            f"raised {quality['false_alarms']} false alarm(s) and missed {quality['missed']}."
        )
        st.write(text)
        if quality["precision"] is not None and quality["recall"] is not None:
            st.caption(
                f"Of the days flagged, {quality['precision']:.0%} were real peaks. "
                f"Of the real peak days, {quality['recall']:.0%} were flagged."
            )

    with st.expander("What the alert levels mean"):
        st.write(
            "The level depends on how far the highest forecast hour of the day is above the "
            "threshold: Low (under 5%), Medium (5% to 10%) and High (10% or more). "
            "The suggested actions below are examples, not tested against real grid operations."
        )
        for level, action in SUGGESTED_ACTION.items():
            st.markdown(f"**{level}:** {action}")

# ---------------------------------------------------------------- about tab
with tab_about:
    st.subheader("About this dashboard")
    st.markdown(
        f"""
This dashboard is part of my project on forecasting Delhi's electricity demand.

**Data.** Delhi's load every 5 minutes (Delhi SLDC data from Kaggle), turned into hourly
averages, and daily temperature and humidity from Open-Meteo.

**Model.** A gradient boosting model that predicts how much the load will differ from the
same hour last week. It uses the hour, weekday, month, weather and recent loads that are at
least a day old, so it works as a day-ahead forecast.

**What you see.** Forecasts for hours the model never saw during training
({meta['start']} onwards). Over the whole period the average error is
{meta['mape_model']}% for the model and {meta['mape_last_week']}% for "same hour last week".

**Limits.**
- The forecasts use the real weather of each day, as if the weather forecast were perfect.
- This is a replay of past data, not a live feed.
- Public holidays and festivals are not modelled.
- The peak alert rules are simple examples and have not been tested on a real grid.

Code and write-up: [{REPO_URL}]({REPO_URL})
"""
    )

"""
demand_response.py
------------------
Simple rule-based peak alerts used by the dashboard.

A "peak hour" is an hour where the forecast load is at or above a threshold chosen
in the dashboard. For every day with peak hours, one alert is created. Its level
depends on how far the highest forecast hour is above the threshold.

These rules are an example of how a forecast could drive demand-response
actions. They are not tuned or checked against real grid operations.
"""

import pandas as pd

SUGGESTED_ACTION = {
    "Low": "Watch. No action yet, check again as the day gets closer.",
    "Medium": "Prepare. Ask large consumers to move non-critical load away from the peak hours and check that backup supply is ready.",
    "High": "Act. Send demand-response requests early, arrange extra supply and keep storage or backup generation ready for the peak hours.",
}


def alert_level(excess_percent: float) -> str:
    """Low below 5% over the threshold, Medium from 5%, High from 10%."""
    if excess_percent >= 10:
        return "High"
    if excess_percent >= 5:
        return "Medium"
    return "Low"


def find_alerts(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per day that has forecast hours at or above the threshold.

    df needs the columns: timestamp, forecast.
    """
    flagged = df[df["forecast"] >= threshold].copy()
    columns = ["Date", "Hours above threshold", "First hour", "Last hour",
               "Peak forecast (MW)", "Peak at", "Above threshold by (%)",
               "Level", "Suggested action"]
    if flagged.empty:
        return pd.DataFrame(columns=columns)

    flagged["date"] = flagged["timestamp"].dt.date
    rows = []
    for date, day in flagged.groupby("date"):
        peak_row = day.loc[day["forecast"].idxmax()]
        excess = (peak_row["forecast"] / threshold - 1) * 100
        level = alert_level(excess)
        rows.append({
            "Date": date,
            "Hours above threshold": len(day),
            "First hour": day["timestamp"].min().strftime("%H:%M"),
            "Last hour": day["timestamp"].max().strftime("%H:%M"),
            "Peak forecast (MW)": round(peak_row["forecast"]),
            "Peak at": peak_row["timestamp"].strftime("%H:%M"),
            "Above threshold by (%)": round(excess, 1),
            "Level": level,
            "Suggested action": SUGGESTED_ACTION[level],
        })
    return pd.DataFrame(rows, columns=columns)


def alert_quality(df: pd.DataFrame, threshold: float) -> dict:
    """Compare days flagged by the forecast with days that really went above
    the threshold. Only days with almost complete actual data are used.

    df needs the columns: timestamp, actual, forecast.
    """
    d = df.copy()
    d["date"] = d["timestamp"].dt.date
    daily = d.groupby("date").agg(
        forecast_peak=("forecast", "max"),
        actual_peak=("actual", "max"),
        actual_hours=("actual", "count"),
    )
    daily = daily[daily["actual_hours"] >= 20].dropna(subset=["forecast_peak", "actual_peak"])

    flagged = daily["forecast_peak"] >= threshold
    real = daily["actual_peak"] >= threshold

    caught = int((flagged & real).sum())
    false_alarms = int((flagged & ~real).sum())
    missed = int((~flagged & real).sum())

    return {
        "days": int(len(daily)),
        "flagged": int(flagged.sum()),
        "real": int(real.sum()),
        "caught": caught,
        "false_alarms": false_alarms,
        "missed": missed,
        "precision": caught / (caught + false_alarms) if (caught + false_alarms) else None,
        "recall": caught / (caught + missed) if (caught + missed) else None,
    }

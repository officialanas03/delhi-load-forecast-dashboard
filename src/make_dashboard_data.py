"""
make_dashboard_data.py
----------------------
Prepares the small data files that the dashboard reads, so the dashboard itself
does not need the raw data or any model training.

What it does:
1. Trains the gradient boosting model on all data BEFORE 1 April 2025.
2. Forecasts every hour from 1 April 2025 to the end of the data, day-ahead style
   (inputs from past load are at least 24 hours old).
3. Saves the forecasts next to the real load and the "same hour last week"
   forecast, so the dashboard can compare them.

Outputs:
- app/forecast_data.csv   hourly actual load and forecasts
- app/metadata.json       dates, model name, default peak threshold, error scores

Like the other scripts, it uses the real weather of each day as if the weather
forecast were perfect.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from baseline_model import load_data
from xgb_model import make_model, build_gb_features, MODEL_NAME

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
APP_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2025-04-01")


def main():
    print(f"Loading merged dataset... (model: {MODEL_NAME})")
    df = load_data()
    y = df["load_MW"]
    last_week = y.shift(168)
    X = build_gb_features(df)

    train = (df.index < START) & y.notna() & last_week.notna()
    print(f"Training on {int(train.sum())} hours before {START.date()}...")
    model = make_model()
    model.fit(X[train], (y - last_week)[train])

    period = df.index >= START
    can_predict = period & last_week.notna()
    forecast = pd.Series(np.nan, index=df.index)
    forecast[can_predict] = last_week[can_predict] + model.predict(X[can_predict])

    out = pd.DataFrame({
        "timestamp": df.index[period],
        "actual": y[period].values,
        "forecast": forecast[period].values,
        "last_week": last_week[period].values,
        "temperature": df["temperature_2m_mean"][period].values,
    })
    number_columns = ["actual", "forecast", "last_week", "temperature"]
    out[number_columns] = out[number_columns].round(1)
    out.to_csv(APP_DIR / "forecast_data.csv", index=False)

    both = out.dropna(subset=["actual", "forecast", "last_week"])
    mape_model = float((both["actual"] - both["forecast"]).abs().div(both["actual"]).mean() * 100)
    mape_naive = float((both["actual"] - both["last_week"]).abs().div(both["actual"]).mean() * 100)

    default_threshold = int(round(y[train].quantile(0.95) / 100) * 100)

    metadata = {
        "model_name": MODEL_NAME,
        "train_end": str(df.index[train].max().date()),
        "start": str(out["timestamp"].min().date()),
        "end": str(out["timestamp"].max().date()),
        "default_threshold": default_threshold,
        "mape_model": round(mape_model, 2),
        "mape_last_week": round(mape_naive, 2),
        "hours_scored": int(len(both)),
    }
    (APP_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nPeriod: {metadata['start']} to {metadata['end']} ({len(out)} hours)")
    print(f"Average error over the whole period: model {metadata['mape_model']}%, "
          f"same hour last week {metadata['mape_last_week']}%")
    print(f"Default peak threshold (95th percentile of training load): {default_threshold} MW")
    print(f"\nSaved -> {APP_DIR / 'forecast_data.csv'}")
    print(f"Saved -> {APP_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()

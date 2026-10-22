"""
xgb_model.py
------------
A stronger day-ahead forecasting model for Delhi's hourly electricity load,
compared against the simple baselines from baseline_model.py.

How the model works:
- It is a gradient boosting model (XGBoost).
- Instead of predicting the load directly, it predicts how much the load will
  differ from the same hour last week, then adds that to last week's value.
  This lets it follow the current load level even when demand grows over time.
- Inputs: hour, day of week, month, day of year, temperature (mean, max, min),
  humidity, temperature change compared with last week, the load 24 hours ago,
  48 hours ago and 168 hours ago, and the average load of the last 7 days.

Day-ahead setting: every input that comes from past load is at least 24 hours
old, so the model could be run at midnight to forecast the whole next day.
Like the baselines, it uses the actual weather of the day as if the weather
forecast were perfect.

Two test periods are used, each trained only on data BEFORE the test period:
- Winter: the last 90 days of the data
- Summer: 15 May to 13 August 2025

Outputs:
- prints a results table for each test period
- reports/model_results.csv
- reports/figures/model_forecast.png            (14 days from each test period)
- reports/figures/model_feature_importance.png  (which inputs matter most)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.linear_model import LinearRegression

from baseline_model import load_data, build_features, score

try:
    from xgboost import XGBRegressor

    MODEL_NAME = "XGBoost"

    def make_model():
        return XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

except Exception as error:  # XGBoost missing or not loading on this computer
    from sklearn.ensemble import HistGradientBoostingRegressor

    MODEL_NAME = "Gradient boosting (scikit-learn)"
    print(f"Note: XGBoost could not be used ({type(error).__name__}). "
          "Using scikit-learn gradient boosting instead.\n")

    def make_model():
        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=5, random_state=42
        )

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DAYS = 14

plt.style.use("seaborn-v0_8-whitegrid")


def build_gb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inputs for the gradient boosting model."""
    y = df["load_MW"]
    idx = df.index
    temp = df["temperature_2m_mean"]

    X = pd.DataFrame(index=idx)
    X["hour"] = idx.hour
    X["day_of_week"] = idx.dayofweek
    X["month"] = idx.month
    X["day_of_year"] = idx.dayofyear
    X["temp_mean"] = temp
    X["temp_max"] = df["temperature_2m_max"]
    X["temp_min"] = df["temperature_2m_min"]
    X["humidity"] = df["relative_humidity_2m_mean"]
    X["temp_change_vs_last_week"] = temp - temp.shift(168)
    X["load_24h_ago"] = y.shift(24)
    X["load_48h_ago"] = y.shift(48)
    X["load_168h_ago"] = y.shift(168)
    X["avg_load_last_7_days"] = y.shift(24).rolling(168, min_periods=120).mean()
    return X


def evaluate_window(df, X_reg, X_gb, start, end):
    """Train on everything before `start`, test between `start` and `end`."""
    y = df["load_MW"]
    last_week = y.shift(168)

    reg_ok = X_reg.notna().all(axis=1)
    train = (df.index < start) & y.notna() & last_week.notna()
    test = (df.index >= start) & (df.index <= end)

    # 1. Same hour last week
    pred_naive = last_week.copy()

    # 2. Linear regression per hour (best of the earlier baselines)
    pred_hourly = pd.Series(np.nan, index=df.index)
    for h in range(24):
        h_train = train & reg_ok & (df.index.hour == h)
        h_test = test & reg_ok & last_week.notna() & (df.index.hour == h)
        if h_train.sum() == 0 or h_test.sum() == 0:
            continue
        m = LinearRegression().fit(X_reg[h_train], y[h_train])
        pred_hourly[h_test] = m.predict(X_reg[h_test])

    # 3. Gradient boosting: predict the change from last week
    model = make_model()
    model.fit(X_gb[train], (y - last_week)[train])
    pred_gb = pd.Series(np.nan, index=df.index)
    can_predict = test & last_week.notna()
    pred_gb[can_predict] = last_week[can_predict] + model.predict(X_gb[can_predict])

    common = (test & y.notna() & pred_naive.notna()
              & pred_hourly.notna() & pred_gb.notna())

    results = pd.DataFrame(
        {
            "Same hour last week": score(y[common], pred_naive[common]),
            "Linear regression per hour": score(y[common], pred_hourly[common]),
            MODEL_NAME: score(y[common], pred_gb[common]),
        }
    ).T.round(2)

    preds = {"naive": pred_naive, "hourly": pred_hourly, "gb": pred_gb}
    return results, preds, model, int(common.sum())


def plot_forecasts(df, windows_out):
    y = df["load_MW"]
    fig, axes = plt.subplots(len(windows_out), 1, figsize=(12, 4.4 * len(windows_out)))
    if len(windows_out) == 1:
        axes = [axes]

    for ax, (name, (start, end, preds)) in zip(axes, windows_out.items()):
        w_start = end - pd.Timedelta(days=PLOT_DAYS)
        w = (df.index > w_start) & (df.index <= end)
        ax.plot(df.index[w], y[w], label="Actual", color="black", linewidth=1.6)
        ax.plot(df.index[w], preds["gb"][w], label=MODEL_NAME, color="tab:green", linewidth=1.3)
        ax.plot(df.index[w], preds["naive"][w], label="Same hour last week",
                color="tab:orange", linewidth=1.0, alpha=0.8)
        ax.set_title(f"{name}: forecasts vs actual load (last {PLOT_DAYS} days of the test period)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Load (MW)")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
        ax.legend(loc="upper left", ncol=3, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_forecast.png", dpi=150)
    plt.close(fig)


def plot_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return False
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    plt.figure(figsize=(8, 5))
    plt.barh(imp.index, imp.values, color="steelblue")
    plt.title(f"What the {MODEL_NAME} model relies on most")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_feature_importance.png", dpi=150)
    plt.close()
    return True


def main():
    print(f"Loading merged dataset... (model: {MODEL_NAME})")
    df = load_data()

    X_reg = build_features(df, include_hour=False)
    X_gb = build_gb_features(df)

    last_time = df.index.max()
    windows = {
        "Winter": (last_time - pd.Timedelta(days=90), last_time),
        "Summer": (pd.Timestamp("2025-05-15"), pd.Timestamp("2025-08-13 23:00")),
    }

    all_results = []
    windows_out = {}
    importance_model = None

    for name, (start, end) in windows.items():
        if start < df.index.min() or end > df.index.max():
            print(f"\nSkipping {name}: dates are outside the data.")
            continue
        results, preds, model, n_hours = evaluate_window(df, X_reg, X_gb, start, end)
        print(f"\n{name} test period: {start.date()} to {end.date()} ({n_hours} hours scored)\n")
        print(results.to_string())

        out = results.copy()
        out.insert(0, "Test period", f"{name} ({start.date()} to {end.date()})")
        all_results.append(out)
        windows_out[name] = (start, end, preds)
        if name == "Winter":
            importance_model = model

    pd.concat(all_results).to_csv(REPORTS_DIR / "model_results.csv")
    plot_forecasts(df, windows_out)
    print(f"\nSaved results -> {REPORTS_DIR / 'model_results.csv'}")
    print(f"Saved chart   -> {FIGURES_DIR / 'model_forecast.png'}")

    if importance_model is not None and plot_importance(importance_model, list(X_gb.columns)):
        print(f"Saved chart   -> {FIGURES_DIR / 'model_feature_importance.png'}")


if __name__ == "__main__":
    main()

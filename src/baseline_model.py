"""
baseline_model.py
-----------------
Builds simple baseline forecasts for Delhi's hourly electricity load and checks
how accurate they are on data the models have not seen.

The three baselines:
1. "Same hour last week": predicts the load at 3 PM Tuesday as the load at 3 PM
   last Tuesday. No learning involved. Any better model should beat this.
2. Linear regression: one model for all hours, learns from hour of day, day of
   week, month, temperature and humidity.
3. Linear regression per hour: a separate small regression for each hour of the
   day (24 models). Load reacts differently to temperature and season at 4 AM
   than at 3 PM, so each hour gets its own settings.

How the test works:
- The last 90 days of data are kept aside as the test period.
- Everything before that is used for training.
- All models are scored on the test period only, on the same hours, and hours
  with missing load data are left out.

Simplification: the regressions use the actual temperature and humidity of each
day, as if the weather forecast were perfect. In real use these would be forecast
values, so real errors would be a little higher.

Outputs:
- prints a small results table
- reports/baseline_results.csv
- reports/figures/baseline_forecast.png  (last 14 days of the test period)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TEST_DAYS = 90
PLOT_DAYS = 14

plt.style.use("seaborn-v0_8-whitegrid")


def load_data() -> pd.DataFrame:
    """Read the merged dataset and make sure there is one row for every hour."""
    path = PROCESSED_DIR / "merged_data.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df.asfreq("h")
    return df


def build_features(df: pd.DataFrame, include_hour: bool = True) -> pd.DataFrame:
    """Turn the calendar and weather columns into inputs for the regression."""
    idx = df.index
    parts = []
    if include_hour:
        parts.append(pd.get_dummies(pd.Series(idx.hour, index=idx), prefix="hour",
                                    drop_first=True, dtype=float))
    parts.append(pd.get_dummies(pd.Series(idx.dayofweek, index=idx), prefix="dow",
                                drop_first=True, dtype=float))
    parts.append(pd.get_dummies(pd.Series(idx.month, index=idx), prefix="month",
                                drop_first=True, dtype=float))

    temp = df["temperature_2m_mean"]
    weather = pd.DataFrame(index=idx)
    weather["temp"] = temp
    # From the load vs temperature chart: load rises above about 24 degrees C and
    # rises a little below about 16 degrees C, so these two columns capture that.
    weather["cooling"] = (temp - 24).clip(lower=0)
    weather["heating"] = (16 - temp).clip(lower=0)
    weather["humidity"] = df["relative_humidity_2m_mean"]
    parts.append(weather)

    return pd.concat(parts, axis=1)


def score(actual: pd.Series, predicted: pd.Series) -> dict:
    error = actual - predicted
    return {
        "MAPE (%)": float((error.abs() / actual).mean() * 100),
        "RMSE (MW)": float(np.sqrt((error ** 2).mean())),
        "MAE (MW)": float(error.abs().mean()),
    }


def main():
    print("Loading merged dataset...")
    df = load_data()
    y = df["load_MW"]

    X_all = build_features(df, include_hour=True)
    X_day = build_features(df, include_hour=False)
    features_ok = X_all.notna().all(axis=1)

    cutoff = df.index.max() - pd.Timedelta(days=TEST_DAYS)
    train_mask = (df.index <= cutoff) & y.notna() & features_ok
    test_mask = (df.index > cutoff) & features_ok

    print(f"  Training: {df.index.min().date()} to {cutoff.date()} ({int(train_mask.sum())} hours)")
    print(f"  Testing:  {(cutoff + pd.Timedelta(hours=1)).date()} to {df.index.max().date()} "
          f"({int(test_mask.sum())} hours)")

    # Baseline 1: same hour last week
    pred_naive = y.shift(168)

    # Baseline 2: one linear regression for all hours
    model = LinearRegression()
    model.fit(X_all[train_mask], y[train_mask])
    pred_lr = pd.Series(np.nan, index=df.index)
    pred_lr[test_mask] = model.predict(X_all[test_mask])

    # Baseline 3: a separate linear regression for each hour of the day
    pred_hourly = pd.Series(np.nan, index=df.index)
    for h in range(24):
        h_train = train_mask & (df.index.hour == h)
        h_test = test_mask & (df.index.hour == h)
        if h_train.sum() == 0 or h_test.sum() == 0:
            continue
        m = LinearRegression()
        m.fit(X_day[h_train], y[h_train])
        pred_hourly[h_test] = m.predict(X_day[h_test])

    # Score all models on exactly the same hours so the comparison is fair
    common = (test_mask & y.notna() & pred_naive.notna()
              & pred_lr.notna() & pred_hourly.notna())
    missing = int(test_mask.sum() - common.sum())
    print(f"  Hours used for scoring: {int(common.sum())} ({missing} left out: missing data)")

    results = pd.DataFrame(
        {
            "Same hour last week": score(y[common], pred_naive[common]),
            "Linear regression": score(y[common], pred_lr[common]),
            "Linear regression per hour": score(y[common], pred_hourly[common]),
        }
    ).T.round(2)

    print("\nResults on the test period:\n")
    print(results.to_string())

    results.to_csv(REPORTS_DIR / "baseline_results.csv")

    # Chart: last PLOT_DAYS days of the test period
    end = df.index.max()
    start = end - pd.Timedelta(days=PLOT_DAYS)
    window = (df.index > start) & (df.index <= end)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(df.index[window], y[window], label="Actual", color="black", linewidth=1.6)
    ax.plot(df.index[window], pred_hourly[window], label="Linear regression per hour",
            color="tab:green", linewidth=1.2)
    ax.plot(df.index[window], pred_naive[window], label="Same hour last week",
            color="tab:orange", linewidth=1.0, alpha=0.8)
    ax.plot(df.index[window], pred_lr[window], label="Linear regression",
            color="tab:blue", linewidth=1.0, alpha=0.7)
    ax.set_title(f"Baseline forecasts vs actual load (last {PLOT_DAYS} days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (MW)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_forecast.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved results -> {REPORTS_DIR / 'baseline_results.csv'}")
    print(f"Saved chart   -> {FIGURES_DIR / 'baseline_forecast.png'}")


if __name__ == "__main__":
    main()

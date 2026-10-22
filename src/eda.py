"""
eda.py
------
Generates exploratory charts from the merged load + weather dataset and
saves them as PNGs in reports/figures/.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def load_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "merged_data.csv"
    return pd.read_csv(path, parse_dates=["timestamp"])


def plot_load_over_time(df):
    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"], df["load_MW"], linewidth=0.5)
    plt.title("Delhi Electricity Load Over Time")
    plt.xlabel("Date")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "load_over_time.png", dpi=150)
    plt.close()


def plot_load_by_hour(df):
    hourly_avg = df.groupby("hour")["load_MW"].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(hourly_avg.index, hourly_avg.values, marker="o")
    plt.title("Average Load by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Load (MW)")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "load_by_hour.png", dpi=150)
    plt.close()


def plot_weekday_vs_weekend(df):
    grouped = df.groupby(["is_weekend", "hour"])["load_MW"].mean().unstack(level=0)
    grouped.columns = ["Weekday", "Weekend"]
    plt.figure(figsize=(8, 4))
    plt.plot(grouped.index, grouped["Weekday"], marker="o", label="Weekday")
    plt.plot(grouped.index, grouped["Weekend"], marker="o", label="Weekend")
    plt.title("Average Load by Hour: Weekday vs Weekend")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Load (MW)")
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "load_weekday_vs_weekend.png", dpi=150)
    plt.close()


def plot_load_vs_temperature(df):
    daily = df.groupby(df["timestamp"].dt.date).agg(
        {"load_MW": "mean", "temperature_2m_mean": "mean"}
    )
    plt.figure(figsize=(7, 5))
    plt.scatter(daily["temperature_2m_mean"], daily["load_MW"], alpha=0.5, s=15)
    plt.title("Daily Average Load vs Temperature")
    plt.xlabel("Mean Temperature (°C)")
    plt.ylabel("Average Load (MW)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "load_vs_temperature.png", dpi=150)
    plt.close()


def plot_load_by_month(df):
    monthly_avg = df.groupby("month")["load_MW"].mean()
    plt.figure(figsize=(8, 4))
    plt.bar(monthly_avg.index, monthly_avg.values, color="steelblue")
    plt.title("Average Load by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Load (MW)")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "load_by_month.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    print("Loading merged dataset...")
    df = load_data()
    print(f"  -> {len(df)} rows")

    print("Generating charts...")
    plot_load_over_time(df)
    plot_load_by_hour(df)
    plot_weekday_vs_weekend(df)
    plot_load_vs_temperature(df)
    plot_load_by_month(df)

    print(f"\nAll charts saved in: {FIGURES_DIR}")

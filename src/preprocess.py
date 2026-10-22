"""
preprocess.py
-------------
Cleans and merges the raw load data and weather data into a single dataset
ready for analysis and modeling.

1. Loads the 5-minute load data and resamples it to hourly averages.
2. Loads the daily weather data.
3. Merges them on date.
4. Adds calendar features: hour, day of week, is_weekend, month.
5. Saves the result to data/processed/merged_data.csv
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_and_resample_load_data() -> pd.DataFrame:
    path = RAW_DIR / "delhi_load_raw.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    hourly = df["load_MW"].resample("1h").mean().reset_index()
    return hourly


def load_weather_data() -> pd.DataFrame:
    path = RAW_DIR / "delhi_weather_raw.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def merge_and_engineer_features(load_df, weather_df):
    load_df["date"] = load_df["timestamp"].dt.date
    weather_df["date"] = weather_df["date"].dt.date

    merged = pd.merge(load_df, weather_df, on="date", how="left")

    merged["hour"] = merged["timestamp"].dt.hour
    merged["day_of_week"] = merged["timestamp"].dt.dayofweek
    merged["is_weekend"] = merged["day_of_week"].isin([5, 6]).astype(int)
    merged["month"] = merged["timestamp"].dt.month

    merged = merged.drop(columns=["date"])
    return merged


if __name__ == "__main__":
    print("Loading and resampling load data to hourly...")
    load_df = load_and_resample_load_data()
    print(f"  -> {len(load_df)} hourly rows")

    print("Loading weather data...")
    weather_df = load_weather_data()
    print(f"  -> {len(weather_df)} daily rows")

    print("Merging and engineering features...")
    merged_df = merge_and_engineer_features(load_df, weather_df)

    out_path = PROCESSED_DIR / "merged_data.csv"
    merged_df.to_csv(out_path, index=False)
    print(f"\nSaved merged dataset -> {out_path}")
    print(f"Shape: {merged_df.shape}")
    print("\nPreview:")
    print(merged_df.head())

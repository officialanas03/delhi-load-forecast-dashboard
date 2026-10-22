"""
fetch_data.py
-------------
Fetches raw data needed for the Delhi load forecasting project:
  1. Historical weather data (Open-Meteo API — free, no key required)
  2. Historical Delhi load/demand data (from Delhi SLDC / Kaggle export)

LOAD DATA NOTE:
Delhi SLDC's live dashboard (https://delhisldc.org) does not offer a clean bulk
historical download via API. The recommended path is:
  1. Download the "Delhi Electricity Load Data (5-min resolution)" dataset from Kaggle:
     https://www.kaggle.com/datasets/prash4nt/delhi-sldc-load-data-5-min-resolution
  2. Place the downloaded CSV into data/raw/delhi_load_raw.csv
This script will then load and do an initial pass over it.

WEATHER DATA:
Open-Meteo's free historical archive API needs no signup. This script fetches daily
temperature (max/min/mean) for Delhi over a given date range.
"""

import pandas as pd
import requests
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Delhi coordinates
LAT, LON = 28.6139, 77.2090


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily historical weather for Delhi from Open-Meteo's archive API.
    Dates in 'YYYY-MM-DD' format.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,relative_humidity_2m_mean",
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["daily"]
    df = pd.DataFrame(data)
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_raw_load_data(filename: str = "delhi_load_raw.csv") -> pd.DataFrame:
    """
    Loads the raw Delhi load CSV (downloaded manually from Kaggle) if present.
    Returns None with an instructional message if not found yet.
    """
    path = RAW_DIR / filename
    if not path.exists():
        print(
            f"[!] {path} not found.\n"
            "    Download the Delhi load dataset from Kaggle and place it here:\n"
            "    https://www.kaggle.com/datasets/prash4nt/delhi-sldc-load-data-5-min-resolution\n"
        )
        return None
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    print("Fetching weather data for Delhi...")
    weather_df = fetch_weather("2019-01-01", "2024-12-31")
    weather_path = RAW_DIR / "delhi_weather_raw.csv"
    weather_df.to_csv(weather_path, index=False)
    print(f"Saved weather data -> {weather_path} ({len(weather_df)} rows)")

    print("\nChecking for load data...")
    load_df = load_raw_load_data()
    if load_df is not None:
        print(f"Loaded load data ({len(load_df)} rows). Preview:")
        print(load_df.head())

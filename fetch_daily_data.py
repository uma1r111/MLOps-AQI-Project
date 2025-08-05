import requests
import pandas as pd
import datetime
import os
import shutil

# ------------------------
# Configuration
# ------------------------

latitude = 24.8607
longitude = 67.0011
timezone = "Asia%2FKarachi"
csv_file = "karachi_weather_apr1_to_current.csv"

# Get yesterday’s date
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
start_date = end_date = yesterday.strftime("%Y-%m-%d")

# ------------------------
# Fetch pollutant data
# ------------------------

pollutant_url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    f"?latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
    f"&timezone={timezone}"
)

print("Fetching pollutant data...")
pollutant_resp = requests.get(pollutant_url)
pollutant_resp.raise_for_status()
pollutant_raw = pollutant_resp.json()["hourly"]

pollutant_df = pd.DataFrame({
    "datetime": pd.to_datetime(pollutant_raw["time"]),
    "pm10": pollutant_raw["pm10"],
    "pm2_5": pollutant_raw["pm2_5"],
    "co": pollutant_raw["carbon_monoxide"],
    "no2": pollutant_raw["nitrogen_dioxide"],
    "so2": pollutant_raw["sulphur_dioxide"],
    "o3": pollutant_raw["ozone"],
    "aqi_us": pollutant_raw["us_aqi"]
})

# ------------------------
# Fetch weather data
# ------------------------

weather_url = (
    "https://api.open-meteo.com/v1/forecast?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
    f"&timezone={timezone}"
)

print("Fetching weather data...")
weather_resp = requests.get(weather_url)
weather_resp.raise_for_status()
weather_raw = weather_resp.json()["hourly"]

weather_df = pd.DataFrame({
    "datetime": pd.to_datetime(weather_raw["time"]),
    "temp_C": weather_raw["temperature_2m"],
    "humidity_%": weather_raw["relative_humidity_2m"],
    "windspeed_kph": weather_raw["wind_speed_10m"],
    "precip_mm": weather_raw["precipitation"]
})

# ------------------------
# Merge data
# ------------------------

merged_df = pd.merge(pollutant_df, weather_df, on="datetime", how="inner")
merged_df["datetime"] = pd.to_datetime(merged_df["datetime"])

print(f"Fetched {len(merged_df)} hourly records for {start_date}")

# ------------------------
# Load existing data & merge
# ------------------------

if os.path.exists(csv_file):
    print("Loading existing CSV...")
    try:
        existing_df = pd.read_csv(csv_file)
        existing_df["datetime"] = pd.to_datetime(existing_df["datetime"], errors="coerce")
    except Exception as e:
        print("Error reading existing file. Aborting to prevent overwrite.")
        raise e

    # Backup original
    # backup_file = csv_file.replace(".csv", "_backup.csv")
    # shutil.copy(csv_file, backup_file)
    # print(f"Backup created at {backup_file}")

    combined_df = pd.concat([existing_df, merged_df], ignore_index=True)
    combined_df.drop_duplicates(subset="datetime", inplace=True)

    # Check if anything changed
    if len(combined_df) == len(existing_df):
        print("No new data added. CSV unchanged.")
    else:
        print(f"CSV updated: {len(combined_df) - len(existing_df)} new rows added.")

    # Safety check
    if len(combined_df) < len(existing_df):
        raise ValueError("Merge resulted in fewer rows than before — aborting write to prevent data loss.")

else:
    print("No existing CSV found. Creating new file.")
    combined_df = merged_df

# ------------------------
# Final write
# ------------------------

combined_df.sort_values("datetime", inplace=True)
combined_df.to_csv(csv_file, index=False)
print(f"Final CSV written with {len(combined_df)} rows.")

# ------------------------
# Save API metrics to metrics/metric.json
# ------------------------

import time
from pathlib import Path
import json

# Record success (both .raise_for_status() passed, so 2 successes)
total_apis = 2
success_count = 2
api_success_rate = round(success_count / total_apis, 2)

# Record response time
avg_response_time = round((pollutant_resp.elapsed.total_seconds() + weather_resp.elapsed.total_seconds()) / 2, 3)

# Prepare metrics directory and path
Path("metrics").mkdir(parents=True, exist_ok=True)
metrics_path = "metrics.json"

# Load existing or start new metrics
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except Exception:
        print("⚠️ Failed to load existing metric.json, starting fresh.")
        metrics = {}
else:
    metrics = {}

# Update only API-related metrics
metrics["api_success_rate"] = api_success_rate
metrics["api_response_time"] = avg_response_time

# Write back to file
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Saved API metrics to {metrics_path}")


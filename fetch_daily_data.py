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
pollutant_data = pollutant_resp.json()["hourly"]
pollutant_df = pd.DataFrame(pollutant_data)
pollutant_df["datetime"] = pd.to_datetime(pollutant_df["time"], utc=True).dt.tz_convert("Asia/Karachi")
pollutant_df.drop(columns=["time"], inplace=True)

# Rename pollutant columns to match CSV
pollutant_df.rename(
    columns={
        "pm10": "pm10",
        "pm2_5": "pm2_5",
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "ozone": "o3",
        "us_aqi": "aqi_us",
    },
    inplace=True,
)

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
weather_data = weather_resp.json()["hourly"]
weather_df = pd.DataFrame(weather_data)
weather_df["datetime"] = pd.to_datetime(weather_df["time"], utc=True).dt.tz_convert("Asia/Karachi")
weather_df.drop(columns=["time"], inplace=True)

# Rename weather columns to match CSV
weather_df.rename(
    columns={
        "temperature_2m": "temp_C",
        "relative_humidity_2m": "humidity_%",
        "wind_speed_10m": "windspeed_kph",
        "precipitation": "precip_mm",
    },
    inplace=True,
)

# ------------------------
# Merge data
# ------------------------

merged_df = pd.merge(pollutant_df, weather_df, on="datetime", how="outer")
merged_df["datetime"] = pd.to_datetime(merged_df["datetime"], utc=True).dt.tz_convert("Asia/Karachi")

print(f"Fetched {len(merged_df)} hourly records for {start_date}")

# ------------------------
# Load existing data & merge
# ------------------------

if os.path.exists(csv_file):
    print("Loading existing CSV...")
    try:
        existing_df = pd.read_csv(csv_file)
        # Parse datetime with explicit timezone handling
        existing_df["datetime"] = pd.to_datetime(existing_df["datetime"], utc=True, errors="coerce").dt.tz_convert("Asia/Karachi")
    except Exception as e:
        print("Error reading existing file. Aborting to prevent overwrite.")
        raise e

    # Ensure no null datetimes
    if existing_df["datetime"].isnull().any():
        print("Warning: Some datetime values in existing CSV are invalid and will be dropped.")
        existing_df = existing_df.dropna(subset=["datetime"])

    # Backup original
    backup_file = csv_file.replace(".csv", "_backup.csv")
    shutil.copy(csv_file, backup_file)
    print(f"Backup created at {backup_file}")

    # Concatenate and handle duplicates
    combined_df = pd.concat([existing_df, merged_df], ignore_index=True)
    # Sort by datetime to ensure consistent deduplication
    combined_df.sort_values("datetime", inplace=True)
    # Keep the latest row for duplicate datetimes
    combined_df.drop_duplicates(subset="datetime", keep="last", inplace=True)

    # Check if anything changed
    if len(combined_df) == len(existing_df):
        print("No new data added. CSV unchanged.")
    else:
        print(f"CSV updated: {len(combined_df) - len(existing_df)} new rows added.")

    # Safety check
    if len(combined_df) < len(existing_df):
        print(f"Warning: Merge resulted in {len(existing_df) - len(combined_df)} fewer rows. Investigating...")
        # Log duplicate datetimes for debugging
        duplicates = combined_df[combined_df["datetime"].duplicated(keep=False)]
        if not duplicates.empty:
            print(f"Found {len(duplicates)} duplicate datetime entries. Check data sources for overlaps.")
        raise ValueError("Merge resulted in fewer rows than before — check for datetime overlaps or data issues.")

else:
    print("No existing CSV found. Creating new file.")
    combined_df = merged_df

# ------------------------
# Final write
# ------------------------

combined_df.sort_values("datetime", inplace=True)
# Ensure consistent column order
column_order = ["datetime", "temp_C", "humidity_%", "windspeed_kph", "precip_mm", "pm10", "pm2_5", "co", "no2", "so2", "o3", "aqi_us"]
combined_df = combined_df[column_order]
combined_df.to_csv(csv_file, index=False)
print(f"Final CSV written with {len(combined_df)} rows.")
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
    f"&hourly=pm10,pm2_5,co,no2,s02,o3,us_aqi"
    f"&timezone={timezone}"
)

print("Fetching pollutant data...")
pollutant_resp = requests.get(pollutant_url)
pollutant_resp.raise_for_status()
pollutant_data = pollutant_resp.json()["hourly"]
pollutant_df = pd.DataFrame(pollutant_data)
pollutant_df["datetime"] = pd.to_datetime(pollutant_df["time"])
pollutant_df.drop(columns=["time"], inplace=True)

# ------------------------
# Fetch weather data
# ------------------------

weather_url = (
    "https://api.open-meteo.com/v1/forecast?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=temp_C,humidity_%,windspeed_kph,precip_mm"
    f"&timezone={timezone}"
)

print("Fetching weather data...")
weather_resp = requests.get(weather_url)
weather_resp.raise_for_status()
weather_data = weather_resp.json()["hourly"]
weather_df = pd.DataFrame(weather_data)
weather_df["datetime"] = pd.to_datetime(weather_df["time"])
weather_df.drop(columns=["time"], inplace=True)

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
        existing_df["datetime"] = pd.to_datetime(existing_df["datetime"], format="ISO8601", errors="coerce")
    except Exception as e:
        print("Error reading existing file. Aborting to prevent overwrite.")
        raise e

    existing_df["datetime"] = pd.to_datetime(existing_df["datetime"])

    # Backup original
    backup_file = csv_file.replace(".csv", "_backup.csv")
    shutil.copy(csv_file, backup_file)
    print(f"Backup created at {backup_file}")

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

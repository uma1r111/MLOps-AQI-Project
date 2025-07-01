import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# --------------------------
# Configuration
# --------------------------

# Karachi coordinates and timezone
latitude = 24.8607
longitude = 67.0011
timezone = "Asia/Karachi"

# File path
csv_file = "karachi_weather_apr1_to_current.csv"

# Calculate yesterday's date
yesterday = datetime.now() - timedelta(days=1)
start_date = end_date = yesterday.strftime("%Y-%m-%d")

# --------------------------
# 1. Fetch pollutant data
# --------------------------

pollution_url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    f"?latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,"
    "sulphur_dioxide,ozone,us_aqi"
    f"&timezone={timezone}"
)

poll_resp = requests.get(pollution_url)
poll_resp.raise_for_status()
poll_data = poll_resp.json()

poll_df = pd.DataFrame({
    "datetime": poll_data["hourly"]["time"],
    "pm10": poll_data["hourly"]["pm10"],
    "pm2_5": poll_data["hourly"]["pm2_5"],
    "co": poll_data["hourly"]["carbon_monoxide"],
    "no2": poll_data["hourly"]["nitrogen_dioxide"],
    "so2": poll_data["hourly"]["sulphur_dioxide"],
    "o3": poll_data["hourly"]["ozone"],
    "aqi_us": poll_data["hourly"]["us_aqi"],
})
poll_df["datetime"] = pd.to_datetime(poll_df["datetime"])

# --------------------------
# 2. Fetch weather data
# --------------------------

weather_url = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=temperature_2m,relative_humidity_2m,"
    "wind_speed_10m,precipitation"
    f"&timezone={timezone}"
)

weather_resp = requests.get(weather_url)
weather_resp.raise_for_status()
weather_data = weather_resp.json()

weather_df = pd.DataFrame({
    "datetime": weather_data["hourly"]["time"],
    "temp_C": weather_data["hourly"]["temperature_2m"],
    "humidity_%": weather_data["hourly"]["relative_humidity_2m"],
    "windspeed_kph": weather_data["hourly"]["wind_speed_10m"],
    "precip_mm": weather_data["hourly"]["precipitation"],
})
weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

# --------------------------
# 3. Merge and reorder
# --------------------------

merged_df = pd.merge(poll_df, weather_df, on="datetime")

# Final column order
final_columns = [
    "datetime", "temp_C", "humidity_%", "windspeed_kph", "precip_mm",
    "pm10", "pm2_5", "co", "no2", "so2", "o3", "aqi_us"
]
merged_df = merged_df[final_columns]

# --------------------------
# 4. Append to CSV
# --------------------------

if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file, parse_dates=["datetime"])
    combined_df = pd.concat([existing_df, merged_df], ignore_index=True)
    combined_df.drop_duplicates(subset="datetime", inplace=True)
else:
    combined_df = merged_df

# Sort and save
combined_df["datetime"] = pd.to_datetime(combined_df["datetime"], errors="coerce")
combined_df = combined_df.dropna(subset=["datetime"])
combined_df.sort_values("datetime", inplace=True)
combined_df.to_csv(csv_file, index=False)

print(f"Appended data for {start_date} to {csv_file}")
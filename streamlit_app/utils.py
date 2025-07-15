import pandas as pd

# GitHub raw URLs
AQI_CSV_URL = "https://raw.githubusercontent.com/uma1r111/10pearls-AQI-Project-/main/bentoml_forecast_output.csv"
WEATHER_CSV_URL = "https://raw.githubusercontent.com/uma1r111/10pearls-AQI-Project-/main/karachi_weather_apr1_to_current.csv"

def load_aqi_data():
    df = pd.read_csv(AQI_CSV_URL, parse_dates=["datetime"])
    df["timestamp"] = df["datetime"].dt.floor("H")  # ← THIS WAS MISSING
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["aqi_category"] = df["predicted_aqi_us"].apply(classify_aqi)
    df.rename(columns={"predicted_aqi_us": "aqi"}, inplace=True)
    return df


def load_weather_data():
    try:
        df = pd.read_csv(WEATHER_CSV_URL, parse_dates=["datetime"])
        df["timestamp"] = df["datetime"].dt.floor("H")  # <--- ROUND TO HOUR
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
        return df
    except Exception as e:
        print(f"Error loading weather data: {e}")
        return pd.DataFrame()


def get_day_data(df, selected_day):
    return df[df["date"] == selected_day].sort_values("timestamp").reset_index(drop=True)

def classify_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_color_for_category(category):
    return {
        "Good": "#009966",
        "Moderate": "#FFDE33",
        "Unhealthy for Sensitive Groups": "#FF9933",
        "Unhealthy": "#CC0033",
        "Very Unhealthy": "#660099",
        "Hazardous": "#7E0023"
    }.get(category, "#000000")

def is_daytime(hour):
    return 6 <= hour < 18
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")

# --- Load dataset ---
raw_csv = "karachi_weather_apr1_to_current.csv"
df = pd.read_csv(raw_csv)

# --- Log transformation ---
skewed_cols = ["co", "pm2_5", "pm10", "precip_mm", "so2", "no2"]
for col in skewed_cols:
    df[f"log_{col}"] = np.log1p(df[col])

# --- Scaling ---
scale_cols = [
    "temp_C", "humidity_%", "windspeed_kph",
    "log_pm2_5", "log_pm10", "log_precip_mm",
    "log_co", "log_no2", "log_so2", "o3"
]
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df[scale_cols]), columns=[f"scaled_{col}" for col in scale_cols])

df_scaled = pd.concat([scaled_df, df[["aqi_us"]].reset_index(drop=True)], axis=1)
df = pd.concat([df.reset_index(drop=True), df_scaled], axis=1)

# --- Lag Features ---
df['aqi_us_lag1'] = df['aqi_us'].shift(1)
df['aqi_us_lag24'] = df['aqi_us'].shift(24)
df['aqi_us_lag12'] = df['aqi_us'].shift(12)

# --- Time-based Features ---
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', dayfirst=True)
df.set_index('datetime', inplace=True)
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['hour_sin'] = np.sin(df['hour'] * 2 * np.pi / 24)
df['hour_cos'] = np.cos(df['hour'] * 2 * np.pi / 24)

# --- Interaction Features ---
df['log_pm2_5_scaled_windspeed_kph'] = df['log_pm2_5'] * df['scaled_windspeed_kph']
df['scaled_temp_C_scaled_o3'] = df['scaled_temp_C'] * df['scaled_o3']
df['scaled_temp_C_scaled_windspeed_kph'] = df['scaled_temp_C'] * df['scaled_windspeed_kph']

# --- Save output ---
final_path = "full_preprocessed_aqi_weather_data_with_all_features.csv"
df.to_csv(final_path, index=True)
print(f"✅ Saved: {final_path} with shape {df.shape}")

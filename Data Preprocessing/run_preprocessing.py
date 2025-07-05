import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")

# --- Load raw updated data ---
raw_csv = "karachi_weather_apr1_to_current.csv"
raw_df = pd.read_csv(raw_csv)
raw_df['datetime'] = pd.to_datetime(raw_df['datetime'], errors="coerce")
raw_df.sort_values("datetime", inplace=True)

# --- Load previous feature-engineered data if exists ---
fe_csv = "full_preprocessed_aqi_weather_data_with_all_features.csv"
if os.path.exists(fe_csv):
    prev_df = pd.read_csv(fe_csv)
    prev_df['datetime'] = pd.to_datetime(prev_df['datetime'], errors="coerce")
    prev_df.sort_values("datetime", inplace=True)
else:
    prev_df = pd.DataFrame(columns=raw_df.columns)

# --- Identify new rows to process ---
new_df = raw_df[~raw_df['datetime'].isin(prev_df['datetime'])].copy()
if new_df.empty:
    print("✅ No new data to process. Skipping.")
    exit()

# --- Log transformation ---
skewed_cols = ["co", "pm2_5", "pm10", "precip_mm", "so2", "no2"]
for col in skewed_cols:
    new_df[f"log_{col}"] = np.log1p(new_df[col])

# --- Scaling ---
scale_cols = [
    "temp_C", "humidity_%", "windspeed_kph",
    "log_pm2_5", "log_pm10", "log_precip_mm",
    "log_co", "log_no2", "log_so2", "o3"
]
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(new_df[scale_cols]), columns=[f"scaled_{col}" for col in scale_cols])

new_df = new_df.reset_index(drop=True)
scaled_df = scaled_df.reset_index(drop=True)
new_df = pd.concat([new_df, scaled_df], axis=1)

# --- Lag Features ---
combined_df = pd.concat([prev_df[['datetime', 'aqi_us']], new_df[['datetime', 'aqi_us']]])
combined_df.sort_values("datetime", inplace=True)
combined_df.set_index("datetime", inplace=True)
combined_df['aqi_us_lag1'] = combined_df['aqi_us'].shift(1)
combined_df['aqi_us_lag12'] = combined_df['aqi_us'].shift(12)
combined_df['aqi_us_lag24'] = combined_df['aqi_us'].shift(24)
combined_df.reset_index(inplace=True)

# Merge lags to new_df
new_df = new_df.merge(combined_df[['datetime', 'aqi_us_lag1', 'aqi_us_lag12', 'aqi_us_lag24']], on='datetime', how='left')

# --- Time-based Features ---
new_df.set_index('datetime', inplace=True)
new_df['hour'] = new_df.index.hour
new_df['day_of_week'] = new_df.index.dayofweek
new_df['is_weekend'] = new_df['day_of_week'].isin([5, 6]).astype(int)
new_df['hour_sin'] = np.sin(new_df['hour'] * 2 * np.pi / 24)
new_df['hour_cos'] = np.cos(new_df['hour'] * 2 * np.pi / 24)

# --- Interaction Features ---
new_df['log_pm2_5_scaled_windspeed_kph'] = new_df['log_pm2_5'] * new_df['scaled_windspeed_kph']
new_df['scaled_temp_C_scaled_o3'] = new_df['scaled_temp_C'] * new_df['scaled_o3']
new_df['scaled_temp_C_scaled_windspeed_kph'] = new_df['scaled_temp_C'] * new_df['scaled_windspeed_kph']

# --- Combine and save final dataset ---
final_df = pd.concat([prev_df, new_df.reset_index()], ignore_index=True)
final_df.drop_duplicates(subset="datetime", keep="last", inplace=True)
final_df.sort_values("datetime", inplace=True)

final_path = "full_preprocessed_aqi_weather_data_with_all_features.csv"
final_df.to_csv(final_path, index=False)
print(f"✅ Saved: {final_path} with shape {final_df.shape}")

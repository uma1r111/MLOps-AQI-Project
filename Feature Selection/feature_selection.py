import pandas as pd
import logging

# Configure logging
logging.basicConfig(filename='feature_selection.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load the full dataset and existing feature selection file
full_df = pd.read_csv("full_preprocessed_aqi_weather_data_with_all_features.csv")
feature_df = pd.read_csv("feature_selection.csv")

# Ensure datetime is in datetime format with correct format
full_df['datetime'] = pd.to_datetime(full_df['datetime'], format='%Y-%m-%d %H:%M:%S')
feature_df['datetime'] = pd.to_datetime(feature_df['datetime'], format='%Y-%m-%d %H:%M:%S')

# Identify new rows based on datetime
last_datetime = feature_df['datetime'].max()
new_data = full_df[full_df['datetime'] > last_datetime]

if not new_data.empty:
    # Select the specified features
    selected_features = ['datetime', 'aqi_us_lag1', 'aqi_us_lag12', 'aqi_us_lag24', 'pm2_5', 'log_pm10', 'scaled_humidity_%', 'scaled_temp_C_scaled_log_windspeed_kph', 'log_so2', 'day_of_week', 'scaled_temp_C', 'scaled_temp_C_scaled_o3', 'log_no2', 'aqi_us']
    new_feature_data = new_data[selected_features]

    # Append new data to existing feature selection file
    updated_df = pd.concat([feature_df, new_feature_data], ignore_index=True).drop_duplicates(subset='datetime', keep='last')

    # Save the updated file
    updated_df.to_csv("feature_selection.csv", index=False)
    logger.info(f"Added {len(new_feature_data)} new rows to feature_selection.csv up to {last_datetime}")
else:
    logger.info("No new data to add to feature_selection.csv")
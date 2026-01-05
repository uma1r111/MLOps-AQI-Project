import pandas as pd
import logging

# Configure logging
logging.basicConfig(filename='feature_selection.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# List of features to keep
selected_features = [
    'datetime', 'aqi_us_lag1', 'aqi_us_lag12', 'aqi_us_lag24', 'pm2_5', 
    'log_pm10', 'scaled_humidity_%', 'scaled_temp_C_scaled_log_windspeed_kph', 
    'log_so2', 'day_of_week', 'scaled_temp_C', 'scaled_temp_C_scaled_o3', 
    'scaled_log_no2', 'scaled_log_so2', 'log_no2', 'aqi_us'
]

try:
    # Load the full dataset and existing feature selection file
    full_df = pd.read_csv("full_preprocessed_aqi_weather_data_with_all_features.csv")
    feature_df = pd.read_csv("feature_selection.csv")

    # Ensure datetime is in datetime format with flexible parsing
    full_df['datetime'] = pd.to_datetime(full_df['datetime'], format='mixed', dayfirst=True, errors='coerce')
    feature_df['datetime'] = pd.to_datetime(feature_df['datetime'], format='mixed', dayfirst=True, errors='coerce')

    # LOGIC: Check if feature_df is empty (only headers exist)
    if feature_df.empty:
        logger.info("feature_selection.csv is empty. Preparing to copy all data.")
        new_data = full_df.copy()
        last_datetime_str = "Initial Load"
    else:
        # If data exists, filter for only new rows
        last_datetime = feature_df['datetime'].max()
        new_data = full_df[full_df['datetime'] > last_datetime]
        last_datetime_str = str(last_datetime)

    # Process if we have data to add
    if not new_data.empty:
        # Select only the specified features from the new data
        new_feature_data = new_data[selected_features]

        # Append new data to existing feature selection file
        updated_df = pd.concat([feature_df, new_feature_data], ignore_index=True)
        
        # Remove duplicates just in case, keeping the newest info
        updated_df = updated_df.drop_duplicates(subset='datetime', keep='last')

        # Save the updated file with consistent datetime format
        updated_df['datetime'] = updated_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        updated_df.to_csv("feature_selection.csv", index=False)
        
        logger.info(f"Success: Added {len(new_feature_data)} rows. (Reference date: {last_datetime_str})")
        print(f"Update Complete: Added {len(new_feature_data)} rows.")
    else:
        logger.info("No new data found to add.")
        print("No new data found.")

except Exception as e:
    logger.error(f"An error occurred: {e}")
    print(f"Error: {e}")
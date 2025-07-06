import pandas as pd
import numpy as np
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(filename='data_quality_report.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load raw CSV
df = pd.read_csv("karachi_weather_apr1_to_current.csv")
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')

# --- Cap Outliers in windspeed_kph ---
max_windspeed = 150  # Realistic maximum for Karachi
if df['windspeed_kph'].max() > max_windspeed:
    original_max = df['windspeed_kph'].max()
    df['windspeed_kph'] = df['windspeed_kph'].clip(upper=max_windspeed)
    logger.info(f"Capped windspeed_kph from {original_max} to {max_windspeed} at {max_windspeed} or below.")

# --- Define Features and Expected Skewness Range for Log Transform ---
features_to_check = ['temp_C', 'humidity_%', 'windspeed_kph', 'precip_mm', 'pm10', 'pm2_5', 'co', 'no2', 'so2', 'o3']  # Exclude aqi_us
expected_log_threshold = 1.0  # Skewness above this triggers log transform
expected_log_features = {'co', 'pm2_5', 'pm10', 'precip_mm', 'so2', 'no2', 'windspeed_kph'}  # Updated to include windspeed_kph

# --- Missing Values Check ---
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100
logger.info("Missing Values (Count, Percentage):")
for col, count in missing_counts.items():
    percent = missing_percent[col]
    logger.info(f"{col}: {count} ({percent:.2f}%)")
    if percent > 10:  # Threshold for warning
        logger.warning(f"High missing data (>10%) in {col}: {percent:.2f}%")
print("-" * 40)

# --- Skewness Check and Range Validation ---
numerical_cols = df.select_dtypes(include=[np.number]).columns
skewness = df[numerical_cols].skew().sort_values(ascending=False)
logger.info("Skewness:")
logger.info(skewness.to_string())
print("-" * 40)

logger.info("Skewness Range Validation (Threshold > 1.0 for log transform):")
log_candidates = set()
for col, skew in skewness.items():
    if col in features_to_check:
        logger.info(f"Checking {col} with skewness {skew:.2f}")
        if skew > expected_log_threshold:
            log_candidates.add(col)
            if col not in expected_log_features:
                logger.warning(f"{col} (skewness: {skew:.2f}) exceeds threshold but not logged. Consider log transform.")
        elif col in expected_log_features and skew <= expected_log_threshold:
            logger.warning(f"{col} (skewness: {skew:.2f}) is below threshold but was logged. Review transform necessity.")
logger.info(f"Features needing log transform (skew > {expected_log_threshold}): {log_candidates}")
print("-" * 40)

# --- Range Validation (Domain Knowledge) ---
expected_ranges = {
    'temp_C': (5, 45),
    'humidity_%': (0, 100),
    'windspeed_kph': (0, 150),
    'precip_mm': (0, 10),
    'pm10': (0, 500),
    'pm2_5': (0, 150),
    'co': (0, 2000),
    'no2': (0, 100),
    'so2': (0, 100),
    'o3': (0, 300),
    'aqi_us': (0, 500),
}
logger.info("Range Validation:")
out_of_range = {}
for col, (min_val, max_val) in expected_ranges.items():
    if col in df.columns:
        out_of_range[col] = ((df[col] < min_val) | (df[col] > max_val)).sum()
        if out_of_range[col] > 0:
            logger.warning(f"{col}: {out_of_range[col]} values out of range [{min_val}, {max_val}]")
logger.info("Range check completed.")
print("-" * 40)

# --- Outlier Detection using IQR ---
logger.info("Outlier Detection (IQR Method):")
outlier_counts = []
for col in numerical_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_counts.append(outliers)
    logger.info(f"{col}: {outliers} outliers")
    if outliers > len(df) * 0.15:  # Threshold of 15% outliers
        logger.warning(f"High outliers (>15%) in {col}: {outliers}")
print("-" * 40)

# --- Datetime Consistency Check ---
df = df.sort_values('datetime')
time_diff = df['datetime'].diff().dropna()
if not time_diff.eq(timedelta(hours=1)).all():
    gaps = time_diff[time_diff != timedelta(hours=1)]
    logger.warning(f"Datetime gaps detected: {gaps}")
else:
    logger.info("Datetime sequence is consistent (1-hour intervals)")
print("-" * 40)

# --- Duplicate Check ---
duplicates = df.duplicated(subset='datetime', keep='first').sum()
if duplicates > 0:
    logger.warning(f"{duplicates} duplicate datetime entries detected")
else:
    logger.info("No duplicate datetime entries")
print("-" * 40)

# --- Quality Status ---
logger.info(f"Quality check conditions: missing_percent.max() > 10: {missing_percent.max() > 10}")
logger.info(f"any(out_of_range.values()): {any(out_of_range.values())}")
logger.info(f"any(outliers > {len(df) * 0.15}): {any(outlier > len(df) * 0.15 for outlier in outlier_counts)}")
logger.info(f"not time_diff.eq(timedelta(hours=1)).all(): {not time_diff.eq(timedelta(hours=1)).all()}")
logger.info(f"duplicates > 0: {duplicates > 0}")
logger.info(f"log_candidates.symmetric_difference(expected_log_features): {log_candidates.symmetric_difference(expected_log_features)}")
quality_issues = any([
    missing_percent.max() > 10,
    any(out_of_range.values()),
    any(outlier > len(df) * 0.15 for outlier in outlier_counts),
    not time_diff.eq(timedelta(hours=1)).all(),
    duplicates > 0,
    log_candidates.symmetric_difference(expected_log_features)
])
if quality_issues:
    logger.error("Data quality check failed. Review warnings in log.")
    print("Data quality check failed. Feature engineering aborted.")
    exit(1)
else:
    logger.info("Data quality check passed. Proceeding with feature engineering.")
    print("Data quality check passed.")

# Optional: Save skewness to file
skewness.to_csv("skewness_report.csv")
logger.info("📁 Skewness report saved to skewness_report.csv")
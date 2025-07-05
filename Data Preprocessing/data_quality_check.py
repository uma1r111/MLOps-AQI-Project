# data_quality_check.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load raw CSV
df = pd.read_csv("karachi_weather_apr1_to_current.csv")

# --- Missing Values Check ---
print("🔍 Missing Values:")
print(df.isnull().sum())
print("-" * 40)

# --- Skewness Check ---
numerical_cols = df.select_dtypes(include=[np.number]).columns
skewness = df[numerical_cols].skew().sort_values(ascending=False)
print("📉 Skewness:")
print(skewness)
print("-" * 40)

# --- Outlier Detection using IQR ---
print("📌 Outlier Detection (IQR Method):")
for col in numerical_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: {outliers} outliers")

print("-" * 40)

# Optional: Save skewness to file
skewness.to_csv("skewness_report.csv")
print("📁 Skewness report saved to skewness_report.csv")

# run_prediction_client.py

import os
import requests
import pandas as pd
import numpy as np
import json
import subprocess

# -----------------------------
# Step 1: Pull latest feature_selection.csv via DVC
# -----------------------------
# print("\n📦 Pulling latest feature_selection.csv from DVC remote (S3)...")
# try:
#     subprocess.run(["dvc", "pull", "feature_selection.csv.dvc"], check=True)
#     print("✅ Pulled feature_selection.csv successfully.")
# except subprocess.CalledProcessError as e:
#     print("❌ Failed to pull feature_selection.csv:", e)
#     exit(1)

# -----------------------------
# Step 2: Load last 72 hours of data
# -----------------------------
print("\n📊 Loading and preparing input data...")
try:
    df = pd.read_csv("feature_selection.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Select exogenous features only
    exog_features = df.drop(columns=["datetime", "aqi_us"])

    # Get last 72 rows
    last_72_exog = exog_features.tail(72).values.tolist()

    if len(last_72_exog) < 72:
        raise ValueError("Insufficient data: Need at least 72 rows of exogenous features.")

    print(f"✅ Prepared {len(last_72_exog)} rows of exogenous input data.")

except Exception as e:
    print("❌ Error while preparing input data:", e)
    exit(1)

# -----------------------------
# Step 3: Format input JSON
# -----------------------------
input_payload = {
    "exog_data": last_72_exog,
    "steps": 72
}

# -----------------------------
# Step 4: Send POST request to BentoML service
# -----------------------------
print("\n🚀 Sending request to BentoML API...")
try:
    response = requests.post(
        url="http://localhost:3000/forecast",
        headers={"Content-Type": "application/json"},
        data=json.dumps(input_payload)
    )

    if response.status_code == 200:
        result = response.json()
        print("✅ Prediction successful.\n")

        # Save to CSV
        pred_df = pd.DataFrame({
            "datetime": result["forecast_dates"],
            "predicted_aqi_us": result["forecast"]
        })
        pred_df.to_csv("bentoml_forecast_output.csv", index=False)
        print("📁 Saved predictions to bentoml_forecast_output.csv")

        # Print summary
        print(f"Average AQI: {np.mean(result['forecast']):.2f}")
        print(f"Min AQI: {np.min(result['forecast']):.2f}")
        print(f"Max AQI: {np.max(result['forecast']):.2f}")

    else:
        print(f"❌ Request failed with status {response.status_code}:", response.text)

except Exception as e:
    print("❌ Error connecting to BentoML service:", e)
    exit(1)

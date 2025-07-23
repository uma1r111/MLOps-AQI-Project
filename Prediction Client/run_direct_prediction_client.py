import os
import pandas as pd
import numpy as np
import bentoml
import subprocess
from datetime import datetime, timedelta
import json

def get_latest_model():
    """Get the latest SARIMAX model from BentoML store"""
    try:
        # Run bentoml models list command
        result = subprocess.run(
            ["bentoml", "models", "list", "--output=json"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        models = json.loads(result.stdout)
        sarimax_models = [m for m in models if m['tag'].startswith('sarimax_model:')]
        
        if not sarimax_models:
            raise ValueError("No SARIMAX models found in BentoML store")
        
        # Get the latest model (sort by tag)
        latest_model = sorted(sarimax_models, key=lambda x: x['tag'])[-1]
        model_tag = latest_model['tag']
        
        print(f"Loading model: {model_tag}")
        model_ref = bentoml.models.get(model_tag)
        model = model_ref.load_model()
        
        return model
        
    except Exception as e:
        print(f"Error loading model from BentoML store: {e}")
        exit(1)

def import_model_if_needed(model_file):
    """Import model only if it doesn't already exist in the store"""
    try:
        # Extract model tag from filename (assuming format: sarimax_model_YYYYMMDD_HHMMSS.bentomodel)
        base_name = os.path.splitext(model_file)[0]  # Remove .bentomodel extension
        expected_tag = base_name  # This should be like "sarimax_model_20241205_143022"
        
        # Check if model already exists
        result = subprocess.run(
            ["bentoml", "models", "list", "--output=json"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        models = json.loads(result.stdout)
        existing_tags = [m['tag'] for m in models]
        
        # Check if any existing tag starts with our expected tag (to handle version suffixes)
        model_exists = any(tag.startswith(expected_tag + ":") for tag in existing_tags)
        
        if model_exists:
            print(f"Model {expected_tag} already exists in BentoML store, skipping import")
            return True
        else:
            print(f"Importing new model: {model_file}")
            subprocess.run(["bentoml", "models", "import", model_file], check=True)
            print(f"Successfully imported model: {model_file}")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"Error importing model: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during model import: {e}")
        return False

# -----------------------------
# Step 1: Import model if needed (called from Jenkins)
# -----------------------------
# Check if we need to import a model (look for .bentomodel files)
bentomodel_files = [f for f in os.listdir('.') if f.endswith('.bentomodel')]
if bentomodel_files:
    print(f"\nFound BentoModel file(s): {bentomodel_files}")
    # Import the first (and presumably only) bentomodel file
    model_file = bentomodel_files[0]
    if not import_model_if_needed(model_file):
        print("Failed to import model")
        exit(1)
else:
    print("\nNo .bentomodel files found in current directory")

# -----------------------------
# Step 2: Load model from BentoML store
# -----------------------------
print("\nLoading SARIMAX model from BentoML store...")
model = get_latest_model()

# -----------------------------
# Step 3: Load and prepare input data
# -----------------------------
print("\nLoading and preparing input data...")
try:
    df = pd.read_csv("feature_selection.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Select exogenous features only
    exog_features = df.drop(columns=["datetime", "aqi_us"])

    # Get last 72 rows for exogenous data
    last_72_exog = exog_features.tail(72)

    if len(last_72_exog) < 72:
        raise ValueError("Insufficient data: Need at least 72 rows of exogenous features.")

    # Get the last timestamp
    last_timestamp = df["datetime"].iloc[-1]

    print(f"Prepared {len(last_72_exog)} rows of exogenous input data.")
    print(f"Last timestamp: {last_timestamp}")

except Exception as e:
    print("Error while preparing input data:", e)
    exit(1)

# -----------------------------
# Step 4: Make direct predictions using loaded model
# -----------------------------
print("\nMaking predictions with loaded model...")
try:
    # Make 72-step forecast
    forecast_steps = 72
    forecast = model.forecast(steps=forecast_steps, exog=last_72_exog)
    
    # Create forecast timestamps
    forecast_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=forecast_steps,
        freq='H'
    )
    
    # Format forecast dates as strings
    forecast_dates = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in forecast_timestamps]
    
    print("Prediction successful.\n")

    # Create results similar to BentoML output
    result = {
        "forecast": forecast.tolist(),
        "forecast_dates": forecast_dates
    }

    # Save to CSV
    pred_df = pd.DataFrame({
        "datetime": result["forecast_dates"],
        "predicted_aqi_us": result["forecast"]
    })
    pred_df.to_csv("direct_forecast_output.csv", index=False)
    print("Saved predictions to direct_forecast_output.csv")

    # Print summary
    print(f"Average AQI: {np.mean(result['forecast']):.2f}")
    print(f"Min AQI: {np.min(result['forecast']):.2f}")
    print(f"Max AQI: {np.max(result['forecast']):.2f}")
    print(f"Forecast periods: {len(result['forecast'])}")
    print(f"Time range: {result['forecast_dates'][0]} to {result['forecast_dates'][-1]}")

except Exception as e:
    print("Error making predictions:", e)
    exit(1)

print("\nDirect prediction completed successfully!")
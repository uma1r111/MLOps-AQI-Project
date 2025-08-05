import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import os
import bentoml
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

# MLflow Configuration
# -------------------------------
MLFLOW_TRACKING_URI = "http://localhost:8000"
EXPERIMENT_NAME = "AQI Model Logging"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()
# ----------------------
# Step 2: Load and Filter Data (matching training code)
# ----------------------
df = pd.read_csv("feature_selection.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# Filter last 90 days from the latest date
latest_date = df["datetime"].max()
start_date = latest_date - pd.Timedelta(days=90)
df = df[df["datetime"] > start_date]
df = df.sort_values("datetime")

print(f"Filtered data rows: {len(df)}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# ----------------------
# Step 3: Prepare data (same as training)
# ----------------------
target_col = "aqi_us"
target = df[target_col]
exog_features = df.drop(columns=["datetime", target_col])

print(f"Target shape: {target.shape}")
print(f"Exogenous features shape: {exog_features.shape}")

# ----------------------
# Step 4: Compare two sets of best parameters
# ----------------------
best_params1 = {
    "p": 0,
    "d": 0,
    "q": 0,
    "P": 2,
    "D": 0,
    "Q": 2,
    "seasonal_period": 6
}

best_params2 = {
    'p': 2, 
    'd': 0, 
    'q': 0, 
    'P': 2, 
    'D': 0, 
    'Q': 2, 
    'seasonal_period': 12
}

# Split data for model comparison (80% train, 20% validation)
train_size = int(len(df) * 0.8)
train_target = target.iloc[:train_size]
val_target = target.iloc[train_size:]
train_exog = exog_features.iloc[:train_size].values
val_exog = exog_features.iloc[train_size:].values

print(f"Training size: {len(train_target)}, Validation size: {len(val_target)}")

models_results = {}

for i, params in enumerate([best_params1, best_params2], 1):
    print(f"\n🔧 Training SARIMAX model {i} with params: {params}")
    
    try:
        # Train model
        model = SARIMAX(
            endog=train_target,
            exog=train_exog,
            order=(params["p"], params["d"], params["q"]),
            seasonal_order=(params["P"], params["D"], params["Q"], params["seasonal_period"]),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=0, maxiter=50)
        
        # Validate model
        val_preds = fitted_model.forecast(steps=len(val_target), exog=val_exog)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from math import sqrt
        
        rmse = sqrt(mean_squared_error(val_target, val_preds))
        mae = mean_absolute_error(val_target, val_preds)
        aic = fitted_model.aic
        
        models_results[f'model_{i}'] = {
            'params': params,
            'fitted_model': fitted_model,
            'rmse': rmse,
            'mae': mae,
            'aic': aic
        }
        
        print(f"   Model {i} trained successfully!")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   AIC: {aic:.4f}")
        
    except Exception as e:
        print(f" Model {i} training failed: {e}")
        continue

# Select best model based on lowest RMSE
if models_results:
    best_model_name = min(models_results.keys(), key=lambda x: models_results[x]['rmse'])
    best_model_info = models_results[best_model_name]
    
    print(f"\n Best model: {best_model_name}")
    print(f"   Parameters: {best_model_info['params']}")
    print(f"   RMSE: {best_model_info['rmse']:.4f}")
    print(f"   MAE: {best_model_info['mae']:.4f}")
    print(f"   AIC: {best_model_info['aic']:.4f}")

    # Start MLflow run
    run_date = datetime.today().strftime("%Y-%m-%d")
    with mlflow.start_run(run_name=f"SARIMAX Run {run_date}"):
        # Log parameters
        mlflow.log_params(best_model_info['params'])

        # Log metrics
        mlflow.log_metrics({
            'rmse': best_model_info['rmse'],
            'mae': best_model_info['mae'],
            'aic': best_model_info['aic']
        })

        mlflow.set_tag("stage", "daily_training")
        mlflow.set_tag("model_type", "SARIMAX")


        # Optional: log the model file as artifact
        import joblib
        model_file = "sarimax_model.pkl"
        joblib.dump(best_model_info['fitted_model'], model_file)
        mlflow.log_artifact(model_file)

        print("✅ Best model logged to MLflow successfully.")
        
    # Retrain best model on full dataset for final predictions
    print(f"\n Retraining best model on full dataset...")
    final_model = SARIMAX(
        endog=target,
        exog=exog_features.values,
        order=(best_model_info['params']["p"], best_model_info['params']["d"], best_model_info['params']["q"]),
        seasonal_order=(best_model_info['params']["P"], best_model_info['params']["D"], best_model_info['params']["Q"], best_model_info['params']["seasonal_period"]),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    fitted_model = final_model.fit(disp=0, maxiter=50)
    print(" Final model trained on full dataset!")

    # Save the trained model with the legacy API
    try:
        from bentoml.picklable_model import save_model
        saved_model = save_model(
            "sarimax_model",
            fitted_model,
            metadata={
                "model_type": "SARIMAX",
                "best_params": best_model_info['params'],
                "rmse": best_model_info['rmse'],
                "mae": best_model_info['mae'],
                "aic": best_model_info['aic'],
                "features": list(exog_features.columns),
                "target": target_col
            }
        )
        print(f" Model saved to BentoML: {saved_model}")
    except Exception as e:
        print(f" Failed to save model: {e}")

else:
    print(" No models trained successfully!")
    exit(1)

# ----------------------
# Step 6: Forecast next 72 hours (3 days)
# ----------------------
PREDICT_HORIZON = 72

# Use last 72 hours of exogenous data for forecasting
last_exog = exog_features.iloc[-PREDICT_HORIZON:].values

print(f"Forecasting next {PREDICT_HORIZON} hours...")
try:
    future_preds = fitted_model.forecast(steps=PREDICT_HORIZON, exog=last_exog)
    
    # Generate future dates starting from last datetime + 1 hour
    future_dates = pd.date_range(
        start=df['datetime'].iloc[-1] + pd.Timedelta(hours=1),
        periods=PREDICT_HORIZON, 
        freq='h'
    )
    
    # Create output dataframe
    output_df = pd.DataFrame({
        "datetime": future_dates,
        "predicted_aqi_us": future_preds
    })
    
    # Save results
    output_df.to_csv("predictions.csv", index=False)
    print("AQI predictions for next 3 days saved to predictions.csv")
    
    # Display summary
    print(f"\nPrediction Summary:")
    print(f"Average predicted AQI: {future_preds.mean():.2f}")
    print(f"Min predicted AQI: {future_preds.min():.2f}")
    print(f"Max predicted AQI: {future_preds.max():.2f}")
    print(f"Prediction period: {future_dates[0]} to {future_dates[-1]}")
    
except Exception as e:
    print(f"Forecasting failed: {e}")
    exit(1)

print("\nPrediction workflow completed successfully!")

# ----------------------
# Step 7: Check if predictions changed
# ----------------------
import hashlib

def file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

prev_hash_file = "predictions.csv.dvc"
new_hash = file_hash("predictions.csv")

# Pull previous hash from DVC if exists
if os.path.exists(prev_hash_file):
    with open(prev_hash_file, "r") as f:
        dvc_contents = f.read()
    if new_hash in dvc_contents:
        print("No change detected in predictions.csv (same hash). DVC push will likely skip.")
    else:
        print("New predictions generated. DVC will track the update.")
else:
    print("First time tracking predictions.csv with DVC.")
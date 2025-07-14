# service.py
import bentoml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from typing import List, Dict, Any
from pydantic import BaseModel
import bentoml.exceptions
import os

# Define a Pydantic model for the forecast output
class ForecastOutput(BaseModel):
    forecast: List[float]
    forecast_dates: List[str]
    status: str

# Load the SARIMAX model using the pickled file (outside service context)
model = None
try:
    model_ref = bentoml.models.get("sarimax_model:latest")
    model_path = model_ref.path
    with open(os.path.join(model_path, "saved_model.pkl"), "rb") as f:
        loaded_object = pickle.load(f)
    if isinstance(loaded_object, dict):
        model = loaded_object.get('model') or loaded_object.get('fitted_model')
        if model is None or not hasattr(model, 'forecast'):
            raise ValueError("No callable SARIMAX model found in the loaded object.")
    else:
        model = loaded_object
    if not hasattr(model, 'forecast'):
        raise ValueError(f"Loaded object {type(model)} does not have a forecast method.")
except bentoml.exceptions.NotFound:
    raise ValueError("SARIMAX model 'sarimax_model:latest' not found. Please save the model first.")
except Exception as e:
    raise ValueError(f"Failed to load SARIMAX model: {str(e)}")

# Create a BentoML service
svc = bentoml.Service("sarimax_forecaster")

# Define API endpoints using @svc.api() decorator
@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def forecast(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Forecast AQI values using SARIMAX model
    
    Args:
        input_data: Dictionary containing exogenous variables, steps, and last_timestamp
        
    Returns:
        Dictionary with forecast values and dates
    """
    try:
        # Convert input to numpy array
        exog_data = input_data.get("exog_data", [])
        exog_array = np.array(exog_data) if exog_data else None
        
        # Ensure steps is provided, default to 72 if not
        steps = input_data.get("steps", 72)
        
        # Get the last timestamp from input (expected format: "2025-07-13 23:00:00")
        last_timestamp_str = input_data.get("last_timestamp", None)
        if not last_timestamp_str:
            raise ValueError("last_timestamp is required in the input data")
        
        last_timestamp = datetime.strptime(last_timestamp_str, "%Y-%m-%d %H:%M:%S")
        
        # Generate forecast dates starting from the next hour
        forecast_dates = [
            (last_timestamp + timedelta(hours=i+1)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(steps)
        ]
        
        # Make predictions using the loaded SARIMAX model
        if exog_array is not None and exog_array.size > 0:
            predictions = model.forecast(steps=steps, exog=exog_array)
        else:
            predictions = model.forecast(steps=steps)
        
        return {
            "forecast": predictions.tolist(),
            "forecast_dates": forecast_dates,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "forecast": [],
            "forecast_dates": [],
            "status": f"error: {str(e)}"
        }

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def predict_simple(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple prediction endpoint that requires only steps
    (uses last known exogenous data from model)
    """
    try:
        steps = input_data.get("steps", 72)
        
        # Make a simple forecast without exogenous variables
        predictions = model.forecast(steps=steps)
        
        # Generate forecast dates starting from current time (for simplicity)
        base_time = datetime.now()
        forecast_dates = [
            (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(1, steps + 1)
        ]
        
        return {
            "forecast": predictions.tolist(),
            "forecast_dates": forecast_dates,
            "steps_requested": steps,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def health_check(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Health check endpoint to verify the service is running
    """
    try:
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
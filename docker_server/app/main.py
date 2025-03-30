from fastapi import FastAPI, HTTPException
import numpy as np
import mlflow.pyfunc
import os
from prometheus_client import start_http_server, Summary, Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi.openapi.utils import get_openapi
import yaml

# set the MLflow Tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

app = FastAPI()

# Create a metric to track in Prometheus
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count', 'Total number of requests')


def load_model(model_name: str, version: int = None):
    """
    Load a model from the MLflow Model Registry.

    Args:
        model_name (str): The name of the model to load.
        version (int, optional): The version of the model to load. If not provided, the latest version is loaded.

    Returns:
        mlflow.pyfunc.PyFuncModel: The loaded MLflow model.

    Raises:
        HTTPException: If the model cannot be loaded.
    """
    try:
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model '{model_name}': {str(e)}")

@app.post("/predict/logistic_regression")
@REQUEST_TIME.time()
@REQUEST_COUNT.count_exceptions()
def predict_logistic_regression(data: dict, version: int = None):
    """
    Serve predictions for the Logistic Regression model.

    Args:
        data (dict): The input data containing features for prediction.
        version (int, optional): The version of the model to use. Defaults to the latest version.

    Returns:
        dict: A dictionary containing the model name, version, and prediction.

    Raises:
        HTTPException: If there is an error during prediction.
    """
    try:
        # Load model
        model = load_model("Logistic Regression", version)

        # Validate input features
        features = np.array(data["features"]).reshape(1, -1)

        # Make predictions
        prediction = model.predict(features)
        return {"model": "Logistic Regression", "version": version or "latest", "prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")

@app.post("/predict/lightgbm")
@REQUEST_TIME.time()
@REQUEST_COUNT.count_exceptions()
def predict_lightgbm(data: dict, version: int = None):
    """
    Serve predictions for the LightGBM model.

    Args:
        data (dict): The input data containing features for prediction.
        version (int, optional): The version of the model to use. Defaults to the latest version.

    Returns:
        dict: A dictionary containing the model name, version, and prediction.

    Raises:
        HTTPException: If there is an error during prediction.
    """
    try:
        # Load model
        model = load_model("LightGBM", version)

        # Validate input features
        features = np.array(data["features"]).reshape(1, -1)

        # Make predictions
        prediction = model.predict(features)
        return {"model": "LightGBM", "version": version or "latest", "prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")

@app.get("/")
def read_root():
    """
    Root endpoint for the API.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the eDreams ODIGEO Classifier API"}

@app.get("/metrics")
def metrics():
    """
    Endpoint for returning the current metrics of the service.

    Returns:
        Response: The current metrics in Prometheus format.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Load OpenAPI specification from YAML file
with open("/app/src/prediction-openapi.yaml", "r") as f:
    openapi_spec = yaml.safe_load(f)

@app.get("/specifications")
def get_specifications():
    """
    Endpoint for returning the OpenAPI specifications.

    Returns:
        dict: The OpenAPI specifications.
    """
    return openapi_spec


# Start up the server to expose the metrics.
start_http_server(9090)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
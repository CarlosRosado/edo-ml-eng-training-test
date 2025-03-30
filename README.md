# Machine Learning Training and Serving Pipeline

This repository showcases refactor, optimize, and productionize a machine learning pipeline. I started with a file called `my-model.ipynb` and transformed it into a modular, scalable, and production-ready solution. Below, I explain the steps I took, the decisions I made, and why I chose specific tools and techniques.

---

## Table of Contents

1. [Overview](#overview)
2. [Code Refactoring](#code-refactoring)
3. [Pipeline Implementation](#pipeline-implementation)
4. [Performance Optimization](#performance-optimization)
5. [Model Optimization](#model-optimization)
6. [Why Kubernetes and MLflow](#why-kubernetes-and-mlflow)
7. [REST API for Model Serving](#rest-api-for-model-serving)
8. [Dockerization](#dockerization)
9. [Prometheus and Probes](#prometheus-and-probes)
10. [How to Run Using Makefile](#how-to-run-using-makefile)
11. [Calling the Serving API and Metrics](#calling-the-serving-api-and-metrics)

---

## Overview

This project includes:
1. A **training pipeline** that is modular and production-ready.
2. A **REST API** built with FastAPI to serve predictions from the trained model.
3. **Dockerized components** for both training and serving.
4. **Prometheus integration** for monitoring, with liveness and readiness probes.
5. **Kubernetes deployment** for scalability and resilience.
6. **MLflow integration** for experiment tracking and model management.
7. Comprehensive **unit tests** to ensure reliability.

---

## Code Refactoring

When I started with `my-model.ipynb`, the code was difficult to read, maintain, and extend. I refactored it to:
1. **Improve readability and maintainability**:
   - I modularized the code into separate files and functions.
   - I followed PEP 8 standards for clean and consistent formatting.
   - I added meaningful comments and docstrings to explain the logic.

2. **Implement error handling**:
   - I added exception handling to gracefully handle invalid inputs, missing files, and runtime errors.

3. **Write unit tests**:
   - I created unit tests for all major components, including data loading, model training, and prediction.

These changes made the code easier to understand, debug, and extend, ensuring it adheres standards.

---

## Pipeline Implementation

I designed a modular pipeline to separate concerns and improve reusability. The pipeline includes:
1. **Data Loading**:
   - A `data_loader.py` module to handle data loading and preprocessing.
2. **Model Training**:
   - A `model_trainer.py` module to train models like Logistic Regression and LightGBM.
3. **Utilities**:
   - A `utils.py` module for common tasks like directory creation and logging.
4. **Pipeline Orchestration**:
   - A `pipeline_mlfow.py` script to orchestrate the entire process, from data loading to model training and evaluation.

This modular design ensures that each component is reusable and easy to maintain.

---

## Performance Optimization

To improve performance, I focused on:
1. **Efficient Data Preprocessing**:
   - I used `StandardScaler` for feature scaling and ensured compatibility with sparse data.
2. **Parallel Processing**:
   - I enabled parallel processing in LightGBM (`n_jobs=-1`) to utilize multiple CPU cores.
3. **Early Stopping**:
   - I implemented early stopping in LightGBM to prevent overfitting and reduce training time.
4. **Reproducibility**:
   - I used `train_test_split` with a fixed random seed to ensure consistent results.

These optimizations made the pipeline faster, scalable, and more efficient.

---

## Model Optimization

To ensure the model performs well, I implemented:
1. **Cross-Validation**:
   - I used `GridSearchCV` to perform cross-validation and find the best hyperparameters.
2. **Hyperparameter Tuning**:
   - I tuned parameters like `learning_rate`, `n_estimators`, and `max_depth` for LightGBM.
3. **Evaluation Metrics**:
   - I used metrics like accuracy and AUC-ROC to evaluate model performance.

These steps ensured the model generalizes well to unseen data and achieves optimal performance.

---

## Why Kubernetes and MLflow

### Why Kubernetes?
I chose Kubernetes because:
1. It allows **scalability** by enabling horizontal scaling of the REST API and training jobs.
2. It ensures **resilience** with self-healing, liveness, and readiness probes.
3. It simplifies **deployment automation** with YAML configuration files.
4. It provides **container orchestration**, ensuring consistent environments across development, staging, and production.

### Why MLflow?
I integrated MLflow because:
1. It tracks experiments, including metrics like accuracy and AUC, in a centralized dashboard.
2. It provides a **model registry** for managing multiple versions of a model.
3. It ensures **reproducibility** by logging hyperparameters, metrics, and artifacts.
4. It integrates seamlessly with Python and libraries like `scikit-learn` and `LightGBM`.

---

## REST API for Model Serving

I built a REST API using FastAPI because it is fast, easy to use, and provides built-in OpenAPI support. The API includes:
1. A `/predict` endpoint to accept input features and return predictions.
2. Liveness and readiness probes for monitoring.
3. Input validation to ensure correct data formats.

This API makes it easy to integrate the model into other systems.

---

## Dockerization

I created two Docker images:
1. **Training Image**:
   - Contains the training pipeline and dependencies.
   - Used for running the training process in a containerized environment.
2. **Serving Image**:
   - Contains the REST API and dependencies.
   - Used for serving predictions in production.

This separation ensures that training and serving processes are isolated and reproducible.

---

## Prometheus and Probes

I integrated Prometheus for monitoring and added liveness and readiness probes:
1. **Prometheus Metrics**:
   - Metrics include request counts, response times, and error rates.
2. **Liveness and Readiness Probes**:
   - Liveness probe checks if the API is running.
   - Readiness probe checks if the model is loaded and ready to serve predictions.

These features ensure the service is reliable and monitorable in production.

---

## How to Run Using Makefile

I created a `Makefile` to automate common tasks. Here are the key commands:

1. **Run the Entire Pipeline**:
   - Use `make all` to build Docker images, push them to the registry, and deploy the pipeline to Kubernetes.
   - Example:
     ```bash
     make all
     ```

2. **Clean Up Resources**:
   - Use `make clean` to delete all Kubernetes resources and clean up the environment.
   - Example:
     ```bash
     make clean
     ```

3. **Run Tests**:
   - Use `make test` to execute all unit tests using `pytest`.
   - Example:
     ```bash
     make test
     ```

---

## Calling the Serving API and Metrics

Once the REST API is deployed, you can interact with it to make predictions, retrieve metrics, and check the health of the service.

### 1. **Prediction Endpoint**
The `/predict` endpoint accepts input features and returns predictions from the trained model.

#### Example Request:
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}'
```

#### Example Response:

```bash
{
  "model": "Logistic Regression",
  "prediction": 1
}
```
### 2. Prometheus Metrics
The /metrics endpoint exposes Prometheus-compatible metrics for monitoring the API.

#### Example Request:

```bash
curl http://localhost:8000/metrics
```
#### Example Response:

```bash
# HELP request_processing_seconds Time spent processing requests
# TYPE request_processing_seconds histogram
request_processing_seconds_bucket{le="0.005"} 10.0
request_processing_seconds_bucket{le="0.01"} 20.0
...
```

### 3. Health Checks
The API includes liveness and readiness probes to monitor the health of the service.

#### Liveness Probe:
Checks if the API is running.

```bash
curl http://localhost:8000/health/live
```

#### Readiness Probe:
Checks if the model is loaded and ready to serve predictions.

```bash
curl http://localhost:8000/health/ready
```

### 4. Results and Monitoring
   - Use the /predict endpoint to test the model with different input features.
   - Monitor the API's performance and health using the /metrics, /health/live, and /health/ready endpoints.
   - Prometheus can scrape the /metrics endpoint to visualize and analyze metrics over time.

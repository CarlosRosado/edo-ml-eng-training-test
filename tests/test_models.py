import pytest
import numpy as np
import os
from sklearn.datasets import make_classification
from src.models.logistic_regression import LogisticRegressionModel
from src.models.lightgbm_model import LightGBMModel
import matplotlib.pyplot as plt

# Define the output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../output")
os.makedirs(OUTPUT_DIR, exist_ok=True)  

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    return X, y

def test_logistic_regression_model(sample_data):
    X, y = sample_data
    model = LogisticRegressionModel()
    model.train(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y), "Predictions should match the number of samples."

    # Save a dummy plot 
    plot_path = os.path.join(OUTPUT_DIR, "logistic_regression_plot.png")
    plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.savefig(plot_path)
    plt.close()
    assert os.path.exists(plot_path), "The plot should be saved in the output directory."

def test_lightgbm_model(sample_data):
    X, y = sample_data
    model = LightGBMModel()
    model.train(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y), "Predictions should match the number of samples."

    # Save a dummy plot to the output directory
    plot_path = os.path.join(OUTPUT_DIR, "lightgbm_plot.png")
    plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.savefig(plot_path)
    plt.close()
    assert os.path.exists(plot_path), "The plot should be saved in the output directory."
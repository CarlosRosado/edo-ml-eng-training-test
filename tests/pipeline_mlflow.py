import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_classification
from src.pipeline.pipeline_mlflow import PipelineMLflow
from src.data.data_loader import DataLoader
from src.models.logistic_regression import LogisticRegressionModel
from src.models.lightgbm_model import LightGBMModel

@pytest.fixture
def mock_data():
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X, y


@pytest.fixture
def mock_pipeline(tmpdir):
    output_dir = tmpdir.mkdir("output")
    return PipelineMLflow(output_dir=str(output_dir))


@patch("src.pipeline.pipeline_mlflow.mlflow")
@patch("src.pipeline.pipeline_mlflow.DataLoader")
@patch("src.pipeline.pipeline_mlflow.LogisticRegressionModel")
@patch("src.pipeline.pipeline_mlflow.LightGBMModel")
def test_pipeline_run(mock_lightgbm, mock_logistic, mock_dataloader, mock_mlflow, mock_data, mock_pipeline):

    X, y = mock_data

    # Mock DataLoader
    mock_dataloader_instance = MagicMock()
    mock_dataloader_instance.load_breast_cancer_data.return_value = (X, y)
    mock_dataloader.return_value = mock_dataloader_instance

    # Mock LogisticRegressionModel
    mock_logistic_instance = MagicMock()
    mock_logistic_instance.predict.return_value = y
    mock_logistic_instance.model.predict_proba.return_value = [[0.5, 0.5]] * len(y)
    mock_logistic.return_value = mock_logistic_instance

    # Mock LightGBMModel
    mock_lightgbm_instance = MagicMock()
    mock_lightgbm_instance.predict.return_value = y
    mock_lightgbm_instance.model.predict_proba.return_value = [[0.5, 0.5]] * len(y)
    mock_lightgbm.return_value = mock_lightgbm_instance

    # Run the pipeline
    mock_pipeline.run()

    mock_dataloader_instance.load_breast_cancer_data.assert_called_once()

    mock_logistic_instance.train.assert_called_once()
    mock_logistic_instance.predict.assert_called_once()
    mock_logistic_instance.model.predict_proba.assert_called_once()

    mock_lightgbm_instance.train.assert_called_once()
    mock_lightgbm_instance.predict.assert_called_once()
    mock_lightgbm_instance.model.predict_proba.assert_called_once()

    assert mock_mlflow.start_run.call_count == 2  # One for each model
    assert mock_mlflow.log_metrics.call_count >= 2  # Metrics logged for both models
    assert mock_mlflow.sklearn.log_model.call_count == 2  # Models logged for both models
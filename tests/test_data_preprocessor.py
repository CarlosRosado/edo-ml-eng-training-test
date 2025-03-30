import pytest
import pandas as pd
from src.data.data_preprocessor import DataPreprocessor

def test_data_preprocessor():
    # Sample data
    data = pd.DataFrame({
        'numerical_feature': [1.0, 2.0, 3.0],
        'categorical_feature': ['A', 'B', 'A']
    })

    # Define features
    numerical_features = ['numerical_feature']
    categorical_features = ['categorical_feature']

    # Initialize and apply DataPreprocessor
    preprocessor = DataPreprocessor(numerical_features, categorical_features)
    transformed_data = preprocessor.fit_transform(data)

    assert transformed_data is not None
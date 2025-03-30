import pytest
from src.data.data_loader import DataLoader

def test_load_breast_cancer_data():
    """Test loading the breast cancer dataset."""
    X, y = DataLoader.load_breast_cancer_data()
    assert not X.empty
    assert len(X) == len(y)
    assert "mean radius" in X.columns  
    assert y.nunique() == 2  
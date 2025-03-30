import os
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from src.models.logistic_regression import LogisticRegressionModel

@pytest.fixture
def test_data(tmpdir):
    """Fixture to generate test data and output directory."""
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    output_dir = tmpdir.mkdir("output")
    return X_train, X_test, y_train, y_test, str(output_dir)

def test_logistic_regression_model(test_data):
    """Test LogisticRegressionModel end-to-end."""
    X_train, X_test, y_train, y_test, output_dir = test_data
    model = LogisticRegressionModel(scale_features=True)
    
    # Train and predict
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test), "Prediction length mismatch."

    # Save confusion matrix
    cm = confusion_matrix(y_test, predictions)
    model.save_confusion_matrix(cm, output_dir)
    cm_path = os.path.join(output_dir, "logistic_regression_cm.png")
    assert os.path.exists(cm_path), "Confusion matrix image was not saved."
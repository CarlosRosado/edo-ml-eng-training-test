import os
import joblib
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelTrainer:
    """
    A class for training, evaluating, and saving machine learning models.

    This class provides methods to train a model, evaluate its performance, and save the trained model
    and evaluation metrics. It is designed to work with models that implement specific methods such as
    `train`, `predict`, and optionally `evaluate_with_auc` or `plot_feature_importance`.

    """

    def __init__(self, model, output_dir):
        """
        Initialize the ModelTrainer with a model and output directory.

        Args:
            model (object): The machine learning model to be trained and evaluated.
            output_dir (str): The directory where models and metrics will be saved.
        """
        self.model = model
        self.output_dir = output_dir
        self.models_dir = os.path.join(self.output_dir, "models")
        self.metrics_dir = os.path.join(self.output_dir, "metrics")

    def train(self, X_train, y_train):
        """
        Train the model using the provided training data.

        Args:
            X_train (pd.DataFrame or np.ndarray): The training feature data.
            y_train (pd.Series or np.ndarray): The training target labels.

        Returns:
            None
        """
        logging.info("Starting training...")
        self.model.train(X_train, y_train)
        logging.info("Training completed successfully.")

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using the provided test data.

        This method calculates accuracy, generates a classification report, and computes the confusion matrix.
        It also saves evaluation metrics such as the confusion matrix and optionally AUC-ROC curves or feature
        importance plots if the model supports these methods.

        Args:
            X_test (pd.DataFrame or np.ndarray): The test feature data.
            y_test (pd.Series or np.ndarray): The test target labels.

        Returns:
            None
        """
        logging.info("Starting evaluation...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Classification Report:\n{report}")
        logging.info(f"Confusion Matrix:\n{cm}")

        # Save evaluation metrics (plots) to the metrics directory
        self.model.save_confusion_matrix(cm, self.metrics_dir)
        if hasattr(self.model, "evaluate_with_auc"):
            self.model.evaluate_with_auc(X_test, y_test, self.metrics_dir)
        if hasattr(self.model, "plot_feature_importance"):
            self.model.plot_feature_importance(self.metrics_dir)

    def save_model(self, file_name):
        """
        Save the trained model to the models directory.

        Args:
            file_name (str): The name of the file to save the model as.

        Returns:
            None
        """
        os.makedirs(self.models_dir, exist_ok=True)
        file_path = os.path.join(self.models_dir, file_name)
        joblib.dump(self.model, file_path)
        logging.info(f"Model saved to {file_path}")
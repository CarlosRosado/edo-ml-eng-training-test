import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.logistic_regression import LogisticRegressionModel
from src.models.lightgbm_model import LightGBMModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set the MLflow tracking URI to the Dockerized MLflow server
mlflow.set_tracking_uri("http://mlflow:5000")


class PipelineMLflow:
    """
    A class for orchestrating a machine learning pipeline with MLflow integration.

    This class handles the end-to-end process of loading data, preprocessing it,
    training machine learning models, evaluating their performance, and logging
    metrics and artifacts to MLflow.

    """

    def __init__(self, output_dir):
        """
        Initialize the PipelineMLflow with an output directory.

        Args:
            output_dir (str): The directory where models and metrics will be saved.
        """
        self.output_dir = os.path.abspath(output_dir)
        self.models_dir = os.path.join(self.output_dir, "models")
        self.metrics_dir = os.path.join(self.output_dir, "metrics")
        self._prepare_directories()

    def _prepare_directories(self):
        """
        Create required directories for saving models and metrics.

        Returns:
            None
        """
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        logger.info(f"Directories prepared: {self.models_dir}, {self.metrics_dir}")

    def run(self):
        """
        Execute the machine learning pipeline.

        This method performs the following steps:
        1. Load the dataset.
        2. Preprocess the data (scaling numerical features and encoding categorical features).
        3. Split the data into training and testing sets.
        4. Train and evaluate a Logistic Regression model.
        5. Train and evaluate a LightGBM model.
        6. Log metrics, confusion matrices, and models to MLflow.

        Returns:
            None
        """
        # Load data
        data_loader = DataLoader()
        X, y = data_loader.load_breast_cancer_data()

        # Define numerical and categorical features
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # Preprocess data
        preprocessor = DataPreprocessor(numerical_features, categorical_features)
        X = preprocessor.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Set MLflow experiment
        mlflow.set_experiment("Model Training Pipeline")

        # Train and evaluate Logistic Regression model
        logger.info("Training Logistic Regression Model...")
        with mlflow.start_run(run_name="Logistic Regression"):
            log_reg_model = LogisticRegressionModel(scale_features=True)
            log_reg_model.train(X_train, y_train)
            y_pred = log_reg_model.predict(X_test)
            y_pred_proba = log_reg_model.model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            cm = confusion_matrix(y_test, y_pred)

            # Log metrics to MLflow
            metrics = {"accuracy": accuracy, "auc": auc}
            mlflow.log_metrics(metrics)

            # Save and log confusion matrix
            cm_path = os.path.join(self.metrics_dir, "logistic_regression_cm.png")
            log_reg_model.save_confusion_matrix(cm, self.metrics_dir, file_name="logistic_regression_cm.png")
            mlflow.log_artifact(cm_path)

            # Log model to MLflow
            mlflow.sklearn.log_model(
                sk_model=log_reg_model.model,
                artifact_path="logistic_regression_model",
                registered_model_name="Logistic Regression",
                input_example=X_test[:5]  # Provide a sample input
            )

        # Train and evaluate LightGBM model
        logger.info("Training LightGBM Model...")
        with mlflow.start_run(run_name="LightGBM"):
            lgb_model = LightGBMModel()
            lgb_model.train(X_train, y_train)
            y_pred = lgb_model.predict(X_test)
            y_pred_proba = lgb_model.model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            cm = confusion_matrix(y_test, y_pred)

            # Log metrics to MLflow
            metrics = {"accuracy": accuracy, "auc": auc}
            mlflow.log_metrics(metrics)

            # Save and log confusion matrix
            cm_path = os.path.join(self.metrics_dir, "lightgbm_cm.png")
            lgb_model.save_confusion_matrix(cm, self.metrics_dir, file_name="lightgbm_cm.png")
            mlflow.log_artifact(cm_path)

            # Log model to MLflow
            mlflow.sklearn.log_model(
                sk_model=lgb_model.model,
                artifact_path="lightgbm_model",
                registered_model_name="LightGBM",
                input_example=X_test[:5]  
            )


if __name__ == "__main__":
    pipeline = PipelineMLflow(output_dir="./output")
    pipeline.run()
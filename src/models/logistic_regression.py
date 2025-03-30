from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import os
import logging
from sklearn.metrics import ConfusionMatrixDisplay

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LogisticRegressionModel:
    """
    A wrapper class for the Logistic Regression classifier.

    This class provides methods for training, predicting, evaluating, and tuning
    a Logistic Regression model. It also includes utilities for saving evaluation metrics
    such as ROC curves and confusion matrices.

    """

    def __init__(self, params=None, scale_features=False):
        """
        Initialize the LogisticRegressionModel with optional hyperparameters and feature scaling.

        Args:
            params (dict, optional): Hyperparameters for the Logistic Regression model.
                If not provided, default parameters are used.
            scale_features (bool, optional): Whether to scale features using StandardScaler.
                Defaults to False.
        """
        self.default_params = {
            "penalty": "l2",
            "C": 1.0,  # Regularization strength
            "solver": "saga",  # Efficient for large datasets
            "random_state": 42,
            "max_iter": 1000,  # Increase max iterations for convergence
            "n_jobs": -1  # Enable parallel processing
        }
        self.params = params if params else self.default_params
        self.model = LogisticRegression(**self.params)
        self.scaler = StandardScaler() if scale_features else None  # Optional feature scaling

    def train(self, X_train, y_train):
        """
        Train the Logistic Regression model.

        Args:
            X_train (pd.DataFrame or np.ndarray): The training feature data.
            y_train (pd.Series or np.ndarray): The training target labels.

        Returns:
            None
        """
        logging.info("Starting Logistic Regression training...")

        # Scale features if scaler is enabled
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)

        # Handle sparse data
        if isinstance(X_train, csr_matrix):
            logging.info("Training with sparse data.")
        self.model.fit(X_train, y_train)

        logging.info("Logistic Regression training completed.")

    def predict(self, X_test):
        """
        Predict using the trained Logistic Regression model.

        Args:
            X_test (pd.DataFrame or np.ndarray): The test feature data.

        Returns:
            np.ndarray: Predicted labels for the test data.
        """
        # Scale features if scaler is enabled
        if self.scaler:
            X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)

    def evaluate_with_auc(self, X_test, y_test, output_dir):
        """
        Evaluate the model with AUC-ROC and save the ROC curve plot.

        Args:
            X_test (pd.DataFrame or np.ndarray): The test feature data.
            y_test (pd.Series or np.ndarray): The test target labels.
            output_dir (str): Directory to save the ROC curve plot.

        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)

        # Scale features if scaler is enabled
        if self.scaler:
            X_test = self.scaler.transform(X_test)

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        logging.info(f"AUC-ROC: {auc}")

        # Save ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(plot_path)
        logging.info(f"ROC curve saved to {plot_path}")
        plt.close()

    def tune_hyperparameters(self, X_train, y_train, param_grid, cv=5):
        """
        Tune hyperparameters using GridSearchCV.

        Args:
            X_train (pd.DataFrame or np.ndarray): The training feature data.
            y_train (pd.Series or np.ndarray): The training target labels.
            param_grid (dict): Dictionary specifying the hyperparameters to tune.
            cv (int, optional): Number of cross-validation folds. Defaults to 5.

        Returns:
            None
        """
        logging.info("Starting hyperparameter tuning...")
        
        # Scale features if scaler is enabled
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        logging.info(f"Best parameters: {grid_search.best_params_}")
        logging.info("Hyperparameter tuning completed.")

    def save_confusion_matrix(self, cm, output_dir, file_name="logistic_regression_cm.png"):
        """
        Save the confusion matrix as an image.

        Args:
            cm (np.ndarray): Confusion matrix to save.
            output_dir (str): Directory to save the confusion matrix image.
            file_name (str, optional): Name of the image file. Defaults to "logistic_regression_cm.png".

        Returns:
            str: The full path of the saved confusion matrix image.
        """
        os.makedirs(output_dir, exist_ok=True)  
        file_path = os.path.join(output_dir, file_name)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(file_path)
        plt.close()
        return file_path  
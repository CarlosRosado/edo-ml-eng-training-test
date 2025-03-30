from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LightGBMModel:
    """
    A wrapper class for the LightGBM classifier.

    This class provides methods for training, predicting, evaluating, and tuning
    a LightGBM model. It also includes utilities for saving evaluation metrics
    such as ROC curves and confusion matrices.

    """

    def __init__(self, params=None):
        """
        Initialize the LightGBMModel with optional hyperparameters.

        Args:
            params (dict, optional): Hyperparameters for the LightGBM model.
                If not provided, default parameters are used.
        """
        self.default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": -1,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1  # Enable parallel processing
        }
        self.params = params if params else self.default_params
        self.model = LGBMClassifier(**self.params)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the LightGBM model with optional early stopping.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training feature data.
            y_train (pd.Series or np.ndarray): Training target labels.
            X_val (pd.DataFrame or np.ndarray, optional): Validation feature data.
            y_val (pd.Series or np.ndarray, optional): Validation target labels.

        Returns:
            None
        """
        logging.info("Starting LightGBM training...")
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="logloss",
                early_stopping_rounds=10,
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)
        logging.info("LightGBM training completed.")

    def predict(self, X_test):
        """
        Predict using the trained LightGBM model.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test feature data.

        Returns:
            np.ndarray: Predicted labels for the test data.
        """
        return self.model.predict(X_test)

    def evaluate_with_auc(self, X_test, y_test, output_dir):
        """
        Evaluate the model with AUC-ROC and save the ROC curve plot.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test feature data.
            y_test (pd.Series or np.ndarray): Test target labels.
            output_dir (str): Directory to save the ROC curve plot.

        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)
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

    def save_confusion_matrix(self, cm, output_dir, file_name="lightgbm_cm.png"):
        """
        Save the confusion matrix as an image.

        Args:
            cm (np.ndarray): Confusion matrix to save.
            output_dir (str): Directory to save the confusion matrix image.
            file_name (str, optional): Name of the image file. Defaults to "lightgbm_cm.png".

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

    def tune_hyperparameters(self, X_train, y_train, param_grid, cv=5):
        """
        Tune hyperparameters using GridSearchCV.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training feature data.
            y_train (pd.Series or np.ndarray): Training target labels.
            param_grid (dict): Dictionary specifying the hyperparameters to tune.
            cv (int, optional): Number of cross-validation folds. Defaults to 5.

        Returns:
            None
        """
        logging.info("Starting hyperparameter tuning...")
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
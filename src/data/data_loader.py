import pandas as pd
from sklearn.datasets import load_breast_cancer

class DataLoader:
    """
    A utility class for loading datasets.

    This class provides methods to load data from various sources, such as CSV files
    or built-in datasets like the breast cancer dataset from scikit-learn.
    """

    @staticmethod
    def load_breast_cancer_data():
        """
        Load the breast cancer dataset from scikit-learn.

        Returns:
            tuple: A tuple containing:
                - X (pd.DataFrame): A DataFrame containing the feature data.
                - y (pd.Series): A Series containing the target labels.
        """
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        return X, y
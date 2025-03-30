from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    """
    A utility class for preprocessing datasets.

    This class provides methods for preprocessing datasets by scaling numerical features
    and encoding categorical variables. It uses scikit-learn's `Pipeline` and `ColumnTransformer`
    to create a reusable preprocessing pipeline.

    Attributes:
        numerical_features (list): A list of column names for numerical features.
        categorical_features (list): A list of column names for categorical features.
        pipeline (Pipeline): A scikit-learn pipeline for preprocessing the dataset.
    """
    def __init__(self, numerical_features, categorical_features):
        """
        Initialize the DataPreprocessor with numerical and categorical features.

        Args:
            numerical_features (list): A list of column names for numerical features.
            categorical_features (list): A list of column names for categorical features.
        """        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.pipeline = self._create_pipeline()

    def _create_pipeline(self):
        """
        Create a preprocessing pipeline for scaling numerical features and encoding categorical features.

        Returns:
            ColumnTransformer: A scikit-learn `ColumnTransformer` object that applies transformations
            to numerical and categorical features.
        """
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        return preprocessor

    def fit(self, X):
        """
        Fit the preprocessing pipeline to the dataset.

        Args:
            X (pd.DataFrame): The input dataset to fit the pipeline on.

        Returns:
            None
        """
        self.pipeline.fit(X)

    def transform(self, X):
        """
        Transform the dataset using the fitted preprocessing pipeline.

        Args:
            X (pd.DataFrame): The input dataset to transform.

        Returns:
            np.ndarray: The transformed dataset as a NumPy array.
        """
        return self.pipeline.transform(X)

    def fit_transform(self, X):
        """
        Fit the preprocessing pipeline to the dataset and transform it.

        Args:
            X (pd.DataFrame): The input dataset to fit and transform.

        Returns:
            np.ndarray: The transformed dataset as a NumPy array.
        """
        return self.pipeline.fit_transform(X)
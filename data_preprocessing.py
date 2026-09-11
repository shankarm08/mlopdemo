import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define feature columns
NUMERICAL_FEATURES = ["age", "monthly_bill", "tenure_months", "support_calls", "usage_hours"]
CATEGORICAL_FEATURES = ["contract_type"]
TARGET_COLUMN = "churn"
DROP_COLUMNS = ["customer_id"]


def load_data(file_path: str = "data/customers.csv") -> pd.DataFrame:
    """Load customer churn dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    df = pd.read_csv(file_path)
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer for numerical and categorical preprocessing.
    - Numerical: impute missing with median, scale with StandardScaler
    - Categorical: impute missing with most frequent, encode with OneHotEncoder
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def prepare_data(
    file_path: str = "data/customers.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load data, separate features and target, and split into train and test sets.
    """
    df = load_data(file_path)

    # Drop ID column if present
    feature_df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])

    if TARGET_COLUMN not in feature_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = feature_df.drop(columns=[TARGET_COLUMN])
    y = feature_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test

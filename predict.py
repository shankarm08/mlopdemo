import os
import joblib
import pandas as pd

DEFAULT_MODEL_PATH = os.path.join("models", "model.joblib")


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load the trained machine learning pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. Please run src/train.py first."
        )
    return joblib.load(model_path)


def predict_single(data: dict, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Run prediction on a single customer dictionary.
    Returns dictionary with churn_prediction (0 or 1) and a descriptive message.
    """
    model = load_model(model_path)
    df = pd.DataFrame([data])
    prediction = int(model.predict(df)[0])

    message = (
        "Customer is likely to churn"
        if prediction == 1
        else "Customer is unlikely to churn"
    )

    return {
        "churn_prediction": prediction,
        "message": message,
    }

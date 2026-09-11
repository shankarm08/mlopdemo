import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data_preprocessing import prepare_data, build_preprocessor


def train_model(
    data_path: str = "data/customers.csv",
    models_dir: str = "models",
    experiment_name: str = "customer-churn",
    c_param: float = 1.0,
    max_iter: int = 1000,
):
    """
    Train a Logistic Regression model within an end-to-end Pipeline,
    track metrics & parameters with MLflow, and save the artifact to models/.
    """
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load and prepare data
    print("Loading and splitting dataset...")
    X_train, X_test, y_train, y_test = prepare_data(data_path)

    # 2. Build Pipeline (Preprocessor + Classifier)
    preprocessor = build_preprocessor()
    classifier = LogisticRegression(C=c_param, max_iter=max_iter, random_state=42)

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # 3. Setup MLflow Experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="logistic_regression_baseline") as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        # 4. Train Model
        print("Training model pipeline...")
        model_pipeline.fit(X_train, y_train)

        # 5. Evaluate Model
        print("Evaluating model...")
        y_pred = model_pipeline.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        # 6. Log parameters & metrics to MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", c_param)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # 7. Log model artifact in MLflow
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            name="model",
            serialization_format="cloudpickle",
            input_example=X_train.iloc[:2],
        )

        # 8. Save model locally to models/model.joblib
        saved_model_path = os.path.join(models_dir, "model.joblib")
        joblib.dump(model_pipeline, saved_model_path)
        print(f"Model saved locally to: {saved_model_path}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "model_path": saved_model_path,
    }


if __name__ == "__main__":
    train_model()

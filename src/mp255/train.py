"""Model training pipeline with MLflow tracking and Model Registry for MP255."""

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

from src.mp255.clean import run_clean_pipeline
from src.mp255.config import load_params


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_pipeline(regressor: Any, num_cols: list, cat_cols: list) -> Pipeline:
    """Build scikit-learn Pipeline with preprocessing and specified regressor."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def train_models() -> Dict[str, Any]:
    """Train multiple models, log metrics & artifacts to MLflow, and register champion."""
    params = load_params()
    raw_train_path = params["data"]["raw_train_path"]
    target_col = params["data"]["target_col"]
    num_cols = params["data"]["numeric_features"]
    cat_cols = params["data"]["categorical_features"]
    exp_name = params["model"]["experiment_name"]
    reg_model_name = params["model"]["registered_model_name"]
    random_state = params["model"].get("random_state", 42)
    test_size = params["model"].get("test_size", 0.2)

    # 1. Clean data & compute raw hash
    cleaned_df = run_clean_pipeline()
    data_sha256 = compute_sha256(raw_train_path)
    print(f"Original train.csv SHA256: {data_sha256}")

    feature_cols = num_cols + cat_cols
    X = cleaned_df[feature_cols]
    y = cleaned_df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. Setup MLflow SQLite tracking
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    # Define candidate models
    candidates = [
        (
            "Ridge_alpha_1.0",
            Ridge(alpha=1.0, random_state=random_state),
            {"model_type": "Ridge", "alpha": 1.0},
        ),
        (
            "RandomForest_n100",
            RandomForestRegressor(n_estimators=100, random_state=random_state),
            {"model_type": "RandomForestRegressor", "n_estimators": 100},
        ),
        (
            "GradientBoosting_n100_lr0.1",
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=random_state),
            {"model_type": "GradientBoostingRegressor", "n_estimators": 100, "learning_rate": 0.1},
        ),
    ]

    best_mae = float("inf")
    best_run_id = None
    best_pipeline = None
    best_run_name = None

    for run_name, regressor, model_params in candidates:
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            pipe = build_pipeline(regressor, num_cols, cat_cols)
            pipe.fit(X_train, y_train)

            train_preds = pipe.predict(X_train)
            val_preds = pipe.predict(X_val)

            train_mae = float(mean_absolute_error(y_train, train_preds))
            val_mae = float(mean_absolute_error(y_val, val_preds))

            # Log parameters & metrics required by handbook
            mlflow.log_param("data_sha256", data_sha256)
            for p_k, p_v in model_params.items():
                mlflow.log_param(p_k, p_v)

            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("mae", val_mae)

            # Log model artifact with cloudpickle
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            print(f"Run {run_name} (ID: {run_id}) -> val_mae: {val_mae:.4f}")

            if val_mae < best_mae:
                best_mae = val_mae
                best_run_id = run_id
                best_pipeline = pipe
                best_run_name = run_name

    print(f"\nBest Model: {best_run_name} with val_mae: {best_mae:.4f} (Run ID: {best_run_id})")

    # 3. Register Champion Model
    model_uri = f"runs:/{best_run_id}/model"
    reg_version = mlflow.register_model(model_uri, reg_model_name)
    client = MlflowClient()
    client.set_registered_model_alias(name=reg_model_name, alias="champion", version=reg_version.version)
    print(f"Model registered as '{reg_model_name}' version {reg_version.version} with alias '@champion'")

    # 4. Save champion model artifact locally for FastAPI service
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    champion_path = models_dir / "model.joblib"
    joblib.dump(best_pipeline, champion_path)
    print(f"Champion model saved to {champion_path}")

    # 5. Update findings.yaml
    update_findings_tracking(
        exp_name=exp_name,
        best_run_id=best_run_id,
        registered_model_name=reg_model_name,
        champion_version=int(reg_version.version),
    )

    return {
        "best_run_id": best_run_id,
        "best_mae": best_mae,
        "champion_version": int(reg_version.version),
        "data_sha256": data_sha256,
    }


def update_findings_tracking(
    exp_name: str, best_run_id: str, registered_model_name: str, champion_version: int
) -> None:
    """Update tracking section in findings.yaml files."""
    for yaml_path in ["findings.yaml", "reports/findings.yaml"]:
        p = Path(yaml_path)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "tracking" not in data:
            data["tracking"] = {}

        data["tracking"]["experiment_name"] = exp_name
        data["tracking"]["best_run_id"] = best_run_id
        data["tracking"]["registered_model_name"] = registered_model_name
        data["tracking"]["champion_version"] = champion_version

        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
        print(f"Updated {yaml_path} with tracking run {best_run_id}")


if __name__ == "__main__":
    train_models()

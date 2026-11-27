"""Retraining pipeline (v2) on later data with promotion rule and holdout inference."""

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

from src.mp255.clean import clean_department
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


def run_retraining() -> Dict[str, Any]:
    """Retrain v2 model on train.csv + first 70% of later.csv, compare on last 30%,

    evaluate promotion rule, update champion in MLflow registry, and predict holdout.
    """
    params = load_params()
    raw_train_path = params["data"]["raw_train_path"]
    cleaned_train_path = params["data"]["cleaned_train_path"]
    raw_later_path = params["data"]["raw_later_path"]
    holdout_path = params["data"]["holdout_features_path"]
    num_cols = params["data"]["numeric_features"]
    cat_cols = params["data"]["categorical_features"]
    target_col = params["data"]["target_col"]
    feature_cols = num_cols + cat_cols

    exp_name = params["model"]["experiment_name"]
    reg_model_name = params["model"]["registered_model_name"]
    random_state = params["model"].get("random_state", 42)
    later_train_ratio = params["model"].get("v2_later_train_ratio", 0.70)
    promotion_threshold = params["model"].get("promotion_threshold", 0.15)

    # 1. Load data
    train_df = pd.read_csv(cleaned_train_path)
    later_df = pd.read_csv(raw_later_path).sort_values(by=["date", "record_id"]).reset_index(drop=True)
    later_df["department"] = later_df["department"].apply(clean_department)

    # Split later into 70% train and 30% check
    split_idx = int(len(later_df) * later_train_ratio)
    later_train = later_df.iloc[:split_idx]
    later_check = later_df.iloc[split_idx:]

    print(f"Later train: {len(later_train)} rows | Later check: {len(later_check)} rows")

    # Combine cleaned train with later train
    combined_train = pd.concat([train_df, later_train], ignore_index=True)
    print(f"Retraining v2 on {len(combined_train)} combined rows")

    # 2. Evaluate v1 on the last 30% check set
    v1_model_path = Path("models/model.joblib")
    v1_model = joblib.load(v1_model_path)
    v1_preds = v1_model.predict(later_check[feature_cols])
    mae_v1_on_check = float(mean_absolute_error(later_check[target_col], v1_preds))
    print(f"MAE v1 on check set (last 30%): {mae_v1_on_check:.4f}")

    # 3. Train v2 model
    data_sha256 = compute_sha256(raw_train_path)
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    v2_regressor = Ridge(alpha=1.0, random_state=random_state)
    v2_pipeline = build_pipeline(v2_regressor, num_cols, cat_cols)

    with mlflow.start_run(run_name="Ridge_v2_retrained") as run:
        v2_run_id = run.info.run_id
        v2_pipeline.fit(combined_train[feature_cols], combined_train[target_col])
        v2_preds = v2_pipeline.predict(later_check[feature_cols])
        mae_v2_on_check = float(mean_absolute_error(later_check[target_col], v2_preds))

        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("version", "v2")
        mlflow.log_param("retrained_on_later_ratio", later_train_ratio)
        mlflow.log_metric("mae_v1_on_check", mae_v1_on_check)
        mlflow.log_metric("mae_v2_on_check", mae_v2_on_check)
        mlflow.log_metric("val_mae", mae_v2_on_check)
        mlflow.log_metric("mae", mae_v2_on_check)

        mlflow.sklearn.log_model(
            v2_pipeline,
            artifact_path="model",
            serialization_format="cloudpickle",
        )
        print(f"MAE v2 on check set: {mae_v2_on_check:.4f} (v2 Run ID: {v2_run_id})")

    # 4. Evaluate Promotion Rule: promote v2 if MAE is at least 15% lower than v1
    relative_reduction = (mae_v1_on_check - mae_v2_on_check) / mae_v1_on_check
    should_promote = relative_reduction >= promotion_threshold
    decision = "promote" if should_promote else "keep"
    print(
        f"Relative MAE reduction: {relative_reduction * 100:.2f}% "
        f"(Threshold: {promotion_threshold * 100:.1f}%) -> Decision: {decision}"
    )

    client = MlflowClient()
    # Register v2
    v2_model_uri = f"runs:/{v2_run_id}/model"
    reg_v2 = mlflow.register_model(v2_model_uri, reg_model_name)
    v2_version = int(reg_v2.version)
    print(f"Registered '{reg_model_name}' Version {v2_version}")

    if should_promote:
        client.set_registered_model_alias(name=reg_model_name, alias="champion", version=v2_version)
        champion_pipeline = v2_pipeline
        champion_version = v2_version
        print(f"Alias '@champion' moved to Version {v2_version}")
        # Save updated champion to models/model.joblib
        joblib.dump(v2_pipeline, v1_model_path)
        print(f"Saved updated champion model to {v1_model_path}")
    else:
        champion_pipeline = v1_model
        champion_version = 1
        print("Model v1 retained as @champion")

    # 5. Generate Holdout Predictions using @champion
    holdout_df = pd.read_csv(holdout_path)
    holdout_features = holdout_df.copy()
    holdout_features["department"] = holdout_features["department"].apply(clean_department)

    predictions = champion_pipeline.predict(holdout_features[feature_cols])
    # Ensure non-negative predictions rounded to 2 decimals
    predictions = np.maximum(0.0, np.round(predictions, 2))

    pred_dir = Path("predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_csv = pred_dir / "holdout_predictions.csv"

    pred_df = pd.DataFrame({
        "record_id": holdout_df["record_id"],
        "prediction": predictions,
    })
    pred_df.to_csv(out_csv, index=False)
    print(f"Saved {len(pred_df)} holdout predictions to {out_csv}")

    return {
        "mae_v1_on_check": round(mae_v1_on_check, 4),
        "mae_v2_on_check": round(mae_v2_on_check, 4),
        "decision": decision,
        "champion_version": champion_version,
        "holdout_count": len(pred_df),
    }


if __name__ == "__main__":
    run_retraining()

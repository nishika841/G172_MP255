"""Data drift monitoring (PSI) and slice-based error analysis for MP255."""

from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import yaml

from src.mp255.clean import clean_department
from src.mp255.config import load_params


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between reference and new distributions."""
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges = np.unique(bin_edges)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = np.maximum(exp_counts / len(expected), 1e-4)
    act_pct = np.maximum(act_counts / len(actual), 1e-4)

    psi_val = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi_val, 4)


def run_monitoring_analysis() -> Dict[str, Any]:
    """Compute PSI across all numeric columns and calculate slice-based MAE by wing."""
    params = load_params()
    cleaned_train_path = params["data"]["cleaned_train_path"]
    raw_later_path = params["data"]["raw_later_path"]
    num_cols = params["data"]["numeric_features"]
    cat_cols = params["data"]["categorical_features"]
    group_col = params["data"]["group_column"]
    target_col = params["data"]["target_col"]
    feature_cols = num_cols + cat_cols

    train_df = pd.read_csv(cleaned_train_path)
    later_df = pd.read_csv(raw_later_path)
    later_df["department"] = later_df["department"].apply(clean_department)

    # 1. Compute PSI for every numeric column
    psi_dict = {}
    for col in num_cols:
        psi_dict[col] = compute_psi(train_df[col].values, later_df[col].values, bins=10)

    most_drifted = max(psi_dict, key=psi_dict.get)
    print("--- PSI Values ---")
    for col, p_val in psi_dict.items():
        print(f"  {col}: {p_val}")
    print(f"Most drifted feature: {most_drifted} (PSI: {psi_dict[most_drifted]})\n")

    # 2. Compute v1 MAE on later.csv grouped by wing
    v1_model = joblib.load("models/model.joblib")
    later_df["v1_pred"] = v1_model.predict(later_df[feature_cols])

    mae_by_group = {}
    for grp_val, grp_data in later_df.groupby(group_col):
        grp_mae = float(mean_absolute_error(grp_data[target_col], grp_data["v1_pred"]))
        mae_by_group[str(grp_val)] = round(grp_mae, 4)

    worst_group = max(mae_by_group, key=mae_by_group.get)
    print("--- MAE by Group (Wing) ---")
    for grp_name, g_mae in mae_by_group.items():
        print(f"  {grp_name}: {g_mae}")
    print(f"Worst served group: {worst_group}\n")

    return {
        "psi": psi_dict,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_group,
    }


def update_findings_monitoring(monitoring_results: Dict[str, Any]) -> None:
    """Update monitoring section in findings.yaml and reports/findings.yaml."""
    for yaml_path in ["findings.yaml", "reports/findings.yaml"]:
        p = Path(yaml_path)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "monitoring" not in data:
            data["monitoring"] = {}

        for k, v in monitoring_results.items():
            data["monitoring"][k] = v

        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
        print(f"Updated {yaml_path} with monitoring metrics")


if __name__ == "__main__":
    results = run_monitoring_analysis()
    update_findings_monitoring(results)

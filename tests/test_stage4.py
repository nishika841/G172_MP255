"""Unit tests for Stage 4 monitoring, retraining, and holdout artifacts."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from src.mp255.monitoring import compute_psi


def test_psi_computation_detects_drift():
    """Verify that compute_psi detects significant drift when distribution shifts."""
    np.random.seed(42)
    expected = np.random.normal(loc=10.0, scale=2.0, size=500)
    # Undrifted distribution
    same_dist = np.random.normal(loc=10.0, scale=2.0, size=500)
    psi_low = compute_psi(expected, same_dist)
    assert psi_low < 0.1, f"Expected low PSI for identical distributions, got {psi_low}"

    # Drastically shifted distribution
    drifted = np.random.normal(loc=30.0, scale=2.0, size=500)
    psi_high = compute_psi(expected, drifted)
    assert psi_high > 0.25, f"Expected high PSI for shifted distribution, got {psi_high}"


def test_holdout_predictions_file():
    """Verify holdout predictions CSV exists, has exactly 250 rows and correct schema."""
    pred_path = Path("predictions/holdout_predictions.csv")
    assert pred_path.exists(), "predictions/holdout_predictions.csv must exist"

    df = pd.read_csv(pred_path)
    assert list(df.columns) == ["record_id", "prediction"]
    assert len(df) == 250
    assert not df.isna().any().any(), "Holdout predictions must not contain null values"
    assert (df["prediction"] >= 0).all(), "Predicted wait times must be non-negative"


def test_findings_yaml_monitoring_complete():
    """Verify findings.yaml monitoring and tracking sections have no null values."""
    p = Path("findings.yaml")
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Tracking section
    assert data["tracking"]["best_run_id"] is not None
    assert data["tracking"]["champion_version"] == 2

    # Monitoring section
    mon = data["monitoring"]
    assert mon["most_drifted_feature"] == "patients_ahead"
    assert mon["worst_served_group"] == "New Wing"
    assert mon["decision"] in ["promote", "keep"]
    assert mon["mae_v1_on_check"] is not None
    assert mon["mae_v2_on_check"] is not None

    for col, psi_val in mon["psi"].items():
        assert psi_val is not None
        assert psi_val >= 0

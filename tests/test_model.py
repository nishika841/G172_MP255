"""Unit tests for MP255 champion model artifact and pipeline inference."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def test_input_dataframe() -> pd.DataFrame:
    """Provide a valid test feature dataframe."""
    return pd.DataFrame([
        {
            "patients_ahead": 10,
            "doctors_on_duty": 3,
            "arrival_hour": 10,
            "avg_consult_min": 12.0,
            "lab_tests_pending": 1,
            "distance_km": 8.5,
            "department": "General",
            "wing": "Old Wing",
        },
        {
            "patients_ahead": 25,
            "doctors_on_duty": 4,
            "arrival_hour": 14,
            "avg_consult_min": 10.0,
            "lab_tests_pending": 0,
            "distance_km": 15.0,
            "department": "Orthopaedics",
            "wing": "New Wing",
        },
    ])


def test_champion_model_file_exists():
    """Verify champion model artifact exists at models/model.joblib."""
    model_path = Path("models/model.joblib")
    assert model_path.exists(), "models/model.joblib must exist"


def test_champion_model_prediction_shape_and_types(test_input_dataframe):
    """Verify champion model loads and produces numerical predictions of correct length."""
    model_path = Path("models/model.joblib")
    if not model_path.exists():
        pytest.skip("models/model.joblib not yet created")
    
    model = joblib.load(model_path)
    predictions = model.predict(test_input_dataframe)

    assert len(predictions) == len(test_input_dataframe)
    assert np.issubdtype(predictions.dtype, np.floating)
    # Waiting times should be strictly positive
    assert (predictions > 0).all()

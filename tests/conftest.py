"""Pytest fixtures for MP255 tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Provide a small synthetic DataFrame simulating train.csv anomalies."""
    data = [
        # Normal row
        {
            "record_id": "REC001",
            "date": "2026-03-01",
            "patients_ahead": 12,
            "doctors_on_duty": 4,
            "arrival_hour": 9,
            "avg_consult_min": 11.5,
            "lab_tests_pending": 1,
            "distance_km": 5.2,
            "department": "Gen.",
            "wing": "Old Wing",
            "wait_min": 35.0,
        },
        # Duplicate of REC001
        {
            "record_id": "REC001",
            "date": "2026-03-01",
            "patients_ahead": 12,
            "doctors_on_duty": 4,
            "arrival_hour": 9,
            "avg_consult_min": 11.5,
            "lab_tests_pending": 1,
            "distance_km": 5.2,
            "department": "Gen.",
            "wing": "Old Wing",
            "wait_min": 35.0,
        },
        # Stuck sensor row
        {
            "record_id": "REC002",
            "date": "2026-05-10",
            "patients_ahead": 20,
            "doctors_on_duty": 3,
            "arrival_hour": 10,
            "avg_consult_min": 14.0,
            "lab_tests_pending": 2,
            "distance_km": 12.0,
            "department": "Pae.",
            "wing": "New Wing",
            "wait_min": 55.0,
        },
        # Clean row with whitespace / casing variant
        {
            "record_id": "REC003",
            "date": "2026-04-15",
            "patients_ahead": 5,
            "doctors_on_duty": 5,
            "arrival_hour": 11,
            "avg_consult_min": 9.0,
            "lab_tests_pending": 0,
            "distance_km": 3.0,
            "department": " Orthopaedics ",
            "wing": "Annexe",
            "wait_min": 22.0,
        },
    ]
    return pd.DataFrame(data)

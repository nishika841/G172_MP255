"""API tests for MP255 FastAPI prediction service using TestClient."""

from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify GET /health returns status 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_predict_valid_payload():
    """Verify POST /predict succeeds on valid input with positive prediction."""
    payload = {
        "patients_ahead": 15,
        "doctors_on_duty": 3,
        "arrival_hour": 10,
        "avg_consult_min": 12.5,
        "lab_tests_pending": 1,
        "distance_km": 6.2,
        "department": "General",
        "wing": "Old Wing",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data or "wait_min" in data
    val = data.get("wait_min", data.get("prediction"))
    assert isinstance(val, (int, float))
    assert val > 0


def test_predict_rule1_rejects_patients_ahead_above_150():
    """Rule 1: patients_ahead above 150 must be rejected with HTTP 422."""
    payload = {
        "patients_ahead": 151,  # Exceeds max 150
        "doctors_on_duty": 4,
        "arrival_hour": 9,
        "avg_consult_min": 10.0,
        "lab_tests_pending": 0,
        "distance_km": 5.0,
        "department": "Paediatrics",
        "wing": "New Wing",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rule2_rejects_avg_consult_min_below_2():
    """Rule 2: avg_consult_min below 2 must be rejected with HTTP 422."""
    payload = {
        "patients_ahead": 20,
        "doctors_on_duty": 4,
        "arrival_hour": 9,
        "avg_consult_min": 1.5,  # Below minimum 2.0
        "lab_tests_pending": 0,
        "distance_km": 5.0,
        "department": "Orthopaedics",
        "wing": "Annexe",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_required_field():
    """Missing required fields must be rejected with HTTP 422."""
    payload = {
        "patients_ahead": 10,
        "doctors_on_duty": 2,
        # missing department and wing
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

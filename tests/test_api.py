"""Tests for the FastAPI model serving layer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def ensure_model_artifacts() -> None:
    if not (PROJECT_ROOT / "model" / "model.pkl").exists():
        subprocess.run([sys.executable, "model/train_model.py"], cwd=PROJECT_ROOT, check=True)


from app.main import app  # noqa: E402


client = TestClient(app)


SAMPLE_PAYLOAD = {
    "customer_id": "TEST-API-001",
    "tenure_months": 18,
    "monthly_charges": 84.5,
    "total_charges": 1521.0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "support_tickets": 4,
    "usage_frequency": 9,
    "satisfaction_score": 5.8,
    "product_plan": "Standard",
    "number_of_services": 3,
}


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "model_available" in payload
    assert payload["model_available"] is True
    assert payload["model_loadable"] is True
    assert payload["detail"]


def test_model_info_endpoint() -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"]
    assert payload["version"]
    assert "features" in payload
    assert "decision_threshold" in payload


def test_prediction_endpoint() -> None:
    response = client.post("/predict", json=SAMPLE_PAYLOAD)

    assert response.status_code == 200
    payload = response.json()
    assert payload["customer_id"] == "TEST-API-001"
    assert 0 <= payload["churn_probability"] <= 1
    assert payload["predicted_class"] in {0, 1}
    assert payload["risk_level"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert payload["recommendation"]
    assert payload["model_version"]
    assert payload["timestamp"]


def test_invalid_input_handling() -> None:
    invalid_payload = SAMPLE_PAYLOAD | {"satisfaction_score": 15}
    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_batch_prediction_endpoint() -> None:
    response = client.post("/batch-predict", json={"customers": [SAMPLE_PAYLOAD, SAMPLE_PAYLOAD]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_customers"] == 2
    assert len(payload["predictions"]) == 2
    assert "model_version" in payload


def test_batch_prediction_rejects_too_many_customers() -> None:
    response = client.post("/batch-predict", json={"customers": [SAMPLE_PAYLOAD] * 1001})

    assert response.status_code == 422


def test_monitoring_endpoint_tracks_prediction_history() -> None:
    prediction_response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert prediction_response.status_code == 200

    monitoring_response = client.get("/monitoring")
    assert monitoring_response.status_code == 200
    payload = monitoring_response.json()
    assert payload["total_predictions"] >= 1
    assert 0 <= payload["average_churn_probability"] <= 1
    assert "risk_distribution" in payload
    assert payload["recent_predictions"]

"""Tests for prediction business logic."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.prediction import ChurnPredictionService
from app.schemas import CustomerFeatures


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def ensure_model_artifacts() -> None:
    if not (PROJECT_ROOT / "model" / "model.pkl").exists():
        subprocess.run([sys.executable, "model/train_model.py"], cwd=PROJECT_ROOT, check=True)


@pytest.fixture()
def sample_customer() -> CustomerFeatures:
    return CustomerFeatures(
        customer_id="TEST-001",
        tenure_months=12,
        monthly_charges=91.25,
        total_charges=1095.0,
        contract_type="Month-to-month",
        payment_method="Electronic check",
        support_tickets=5,
        usage_frequency=8,
        satisfaction_score=5.2,
        product_plan="Standard",
        number_of_services=3,
    )


def test_prediction_response_structure(sample_customer: CustomerFeatures) -> None:
    service = ChurnPredictionService()
    prediction = service.predict(sample_customer)

    assert prediction.customer_id == "TEST-001"
    assert 0 <= prediction.churn_probability <= 1
    assert prediction.predicted_class in {0, 1}
    assert prediction.risk_level in {"Low Risk", "Medium Risk", "High Risk"}
    assert prediction.recommendation
    assert prediction.suggested_retention_action
    assert prediction.model_version


def test_feature_payload_excludes_customer_id(sample_customer: CustomerFeatures) -> None:
    payload = sample_customer.feature_payload()

    assert "customer_id" not in payload
    assert list(payload.keys()) == [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "contract_type",
        "payment_method",
        "support_tickets",
        "usage_frequency",
        "satisfaction_score",
        "product_plan",
        "number_of_services",
    ]


def test_model_info_contains_threshold_and_feature_contract() -> None:
    service = ChurnPredictionService()
    info = service.get_model_info()

    assert 0 < info["decision_threshold"] < 1
    assert info["features"] == [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "contract_type",
        "payment_method",
        "support_tickets",
        "usage_frequency",
        "satisfaction_score",
        "product_plan",
        "number_of_services",
    ]


"""Pydantic schemas for API validation and response contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ContractType = Literal["Month-to-month", "One year", "Two year"]
PaymentMethod = Literal["Electronic check", "Credit card", "Bank transfer", "Mailed check"]
ProductPlan = Literal["Basic", "Standard", "Premium", "Enterprise"]
RiskLevel = Literal["Low Risk", "Medium Risk", "High Risk"]


FEATURE_COLUMNS = [
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


class CustomerFeatures(BaseModel):
    """Validated customer profile accepted by the prediction API."""

    customer_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Optional customer identifier for tracing predictions.",
        examples=["CUST-10291"],
    )
    tenure_months: int = Field(..., ge=0, le=120, examples=[18])
    monthly_charges: float = Field(..., ge=0, le=500, examples=[84.5])
    total_charges: float = Field(..., ge=0, le=50000, examples=[1521.0])
    contract_type: ContractType = Field(..., examples=["Month-to-month"])
    payment_method: PaymentMethod = Field(..., examples=["Electronic check"])
    support_tickets: int = Field(..., ge=0, le=50, examples=[4])
    usage_frequency: int = Field(
        ...,
        ge=0,
        le=100,
        description="Approximate number of product sessions in the last 30 days.",
        examples=[9],
    )
    satisfaction_score: float = Field(..., ge=1, le=10, examples=[5.8])
    product_plan: ProductPlan = Field(..., examples=["Standard"])
    number_of_services: int = Field(..., ge=1, le=20, examples=[3])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "CUST-10291",
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
        }
    )

    def feature_payload(self) -> dict[str, object]:
        """Return only model feature columns in the expected order."""

        values = self.model_dump()
        return {column: values[column] for column in FEATURE_COLUMNS}


class PredictionResponse(BaseModel):
    """Single prediction response returned by the API."""

    customer_id: str | None
    churn_probability: float
    predicted_class: int
    risk_level: RiskLevel
    recommendation: str
    suggested_retention_action: str
    model_version: str
    timestamp: datetime


class BatchPredictionRequest(BaseModel):
    """Batch prediction request contract."""

    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response contract."""

    predictions: list[PredictionResponse]
    total_customers: int
    model_version: str
    timestamp: datetime


class HealthResponse(BaseModel):
    """Service health response."""

    status: Literal["healthy", "degraded"]
    model_available: bool
    model_loadable: bool
    environment: str
    detail: str
    timestamp: datetime

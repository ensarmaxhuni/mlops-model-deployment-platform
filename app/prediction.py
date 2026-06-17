"""Model loading and prediction logic."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import settings
from app.logger import get_logger
from app.schemas import CustomerFeatures, FEATURE_COLUMNS, PredictionResponse


logger = get_logger(__name__)


RECOMMENDATIONS = {
    "Low Risk": {
        "recommendation": "Maintain engagement through standard customer success communication.",
        "action": "Continue regular onboarding, product education, and quarterly value check-ins.",
    },
    "Medium Risk": {
        "recommendation": "Offer proactive support and personalized retention outreach.",
        "action": "Schedule a customer success review and recommend plan-fit improvements.",
    },
    "High Risk": {
        "recommendation": "Prioritize immediate retention action, account review, and incentive offer.",
        "action": "Escalate to retention team, review open issues, and prepare a targeted incentive.",
    },
}


class ModelArtifactError(RuntimeError):
    """Raised when model artifacts are missing or invalid."""


class ChurnPredictionService:
    """Loads the trained model and produces business-ready predictions."""

    def __init__(
        self,
        model_path: Path = settings.model_path,
        info_path: Path = settings.model_info_path,
        metrics_path: Path = settings.metrics_path,
    ) -> None:
        self.model_path = model_path
        self.info_path = info_path
        self.metrics_path = metrics_path
        self._model: Any | None = None
        self._model_info: dict[str, Any] | None = None
        self._metrics: dict[str, Any] | None = None

    @property
    def model_loaded(self) -> bool:
        """Return whether the model is currently loaded into memory."""

        return self._model is not None

    def is_model_available(self) -> bool:
        """Return whether the required model artifacts exist."""

        return self.model_path.exists() and self.info_path.exists() and self.metrics_path.exists()

    def load(self) -> None:
        """Load model artifacts from disk if they have not already been loaded."""

        if self._model is not None:
            return

        missing = [
            str(path)
            for path in (self.model_path, self.info_path, self.metrics_path)
            if not path.exists()
        ]
        if missing:
            raise ModelArtifactError(
                "Missing model artifact(s). Run `python model/train_model.py` first: "
                + ", ".join(missing)
            )

        logger.info("Loading model artifact from %s", self.model_path)
        self._model = joblib.load(self.model_path)
        self._model_info = self._read_json(self.info_path)
        self._metrics = self._read_json(self.metrics_path)
        self._validate_feature_contract()

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata."""

        self.load()
        return dict(self._model_info or {})

    def get_metrics(self) -> dict[str, Any]:
        """Return model performance metrics."""

        self.load()
        return dict(self._metrics or {})

    def predict(self, customer: CustomerFeatures) -> PredictionResponse:
        """Generate a churn prediction for one customer."""

        self.load()
        frame = self._to_feature_frame(customer)
        probability = float(self._model.predict_proba(frame)[0][1])
        threshold = float((self._model_info or {}).get("decision_threshold", 0.5))
        predicted_class = int(probability >= threshold)
        risk_level = self._risk_level(probability)
        guidance = RECOMMENDATIONS[risk_level]

        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=round(probability, 4),
            predicted_class=predicted_class,
            risk_level=risk_level,
            recommendation=guidance["recommendation"],
            suggested_retention_action=guidance["action"],
            model_version=str((self._model_info or {}).get("version", "unknown")),
            timestamp=datetime.now(timezone.utc),
        )

    def predict_many(self, customers: list[CustomerFeatures]) -> list[PredictionResponse]:
        """Generate predictions for multiple customers."""

        return [self.predict(customer) for customer in customers]

    def _to_feature_frame(self, customer: CustomerFeatures) -> pd.DataFrame:
        payload = customer.feature_payload()
        missing = [column for column in FEATURE_COLUMNS if column not in payload]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")
        return pd.DataFrame([{column: payload[column] for column in FEATURE_COLUMNS}])

    def _validate_feature_contract(self) -> None:
        info_features = (self._model_info or {}).get("features")
        if info_features != FEATURE_COLUMNS:
            raise ModelArtifactError(
                "Model feature contract mismatch. Expected "
                f"{FEATURE_COLUMNS}, received {info_features}."
            )

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability <= 0.35:
            return "Low Risk"
        if probability <= 0.70:
            return "Medium Risk"
        return "High Risk"

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)


prediction_service = ChurnPredictionService()

"""FastAPI backend for the MLOps Model Deployment Platform."""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.logger import get_logger
from app.prediction import ModelArtifactError, prediction_service
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)


logger = get_logger(__name__)

app = FastAPI(
    title="MLOps Model Deployment Platform",
    description=(
        "Production-style customer churn model serving API with model metadata, "
        "performance metrics, batch prediction, and monitoring summaries."
    ),
    version=settings.api_version,
    contact={"name": "Ensar Maxhuni"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=settings.cors_origin_list != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_history: deque[dict[str, Any]] = deque(maxlen=settings.prediction_history_limit)


@app.exception_handler(ModelArtifactError)
async def model_artifact_exception_handler(_: Request, exc: ModelArtifactError) -> JSONResponse:
    """Convert artifact errors into clear API responses."""

    logger.exception("Model artifact error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": str(exc),
            "action": "Run `python model/train_model.py` to create model artifacts.",
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Return a controlled response for unexpected errors."""

    logger.exception("Unhandled API error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal service error. Check API logs for diagnostics."},
    )


@app.get("/", tags=["Platform"])
def root() -> dict[str, str]:
    """Return platform status and documentation links."""

    return {
        "service": settings.project_name,
        "status": "running",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Platform"])
def health() -> HealthResponse:
    """Return service health."""

    model_available = prediction_service.is_model_available()
    model_loadable = False
    detail = "Model artifacts are missing. Run `python model/train_model.py`."

    if model_available:
        try:
            prediction_service.load()
            model_loadable = prediction_service.model_loaded
            detail = "Model artifacts are present and loadable."
        except Exception as exc:  # Health should report degraded instead of failing hard.
            logger.warning("Model health check failed: %s", exc)
            detail = f"Model artifacts are present but not loadable: {exc}"

    return HealthResponse(
        status="healthy" if model_available and model_loadable else "degraded",
        model_available=model_available,
        model_loadable=model_loadable,
        environment=settings.app_env,
        detail=detail,
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/model-info", tags=["Model"])
def model_info() -> dict[str, Any]:
    """Return model metadata and feature contract."""

    return prediction_service.get_model_info()


@app.get("/metrics", tags=["Model"])
def metrics() -> dict[str, Any]:
    """Return model performance metrics."""

    return prediction_service.get_metrics()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Predict churn risk for a single customer profile."""

    started = perf_counter()
    prediction = prediction_service.predict(customer)
    latency_ms = (perf_counter() - started) * 1000
    _record_prediction(prediction, latency_ms)
    logger.info(
        "Prediction completed customer_id=%s risk=%s probability=%.4f latency_ms=%.2f",
        prediction.customer_id,
        prediction.risk_level,
        prediction.churn_probability,
        latency_ms,
    )
    return prediction


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
def batch_predict(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict churn risk for a batch of customer profiles."""

    started = perf_counter()
    predictions = prediction_service.predict_many(payload.customers)
    total_latency_ms = (perf_counter() - started) * 1000
    per_prediction_latency = total_latency_ms / max(len(predictions), 1)
    for prediction in predictions:
        _record_prediction(prediction, per_prediction_latency)

    model_version = predictions[0].model_version if predictions else "unknown"
    logger.info(
        "Batch prediction completed count=%s total_latency_ms=%.2f",
        len(predictions),
        total_latency_ms,
    )
    return BatchPredictionResponse(
        predictions=predictions,
        total_customers=len(predictions),
        model_version=model_version,
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/monitoring", tags=["Monitoring"])
def monitoring_summary() -> dict[str, Any]:
    """Return in-memory monitoring statistics for portfolio observability demos."""

    if not prediction_history:
        return {
            "total_predictions": 0,
            "average_churn_probability": 0.0,
            "high_risk_customers": 0,
            "average_response_time_ms": 0.0,
            "risk_distribution": {},
            "recent_predictions": [],
        }

    probabilities = [item["churn_probability"] for item in prediction_history]
    latencies = [item["response_time_ms"] for item in prediction_history]
    risk_counts = Counter(item["risk_level"] for item in prediction_history)
    return {
        "total_predictions": len(prediction_history),
        "average_churn_probability": round(sum(probabilities) / len(probabilities), 4),
        "high_risk_customers": risk_counts.get("High Risk", 0),
        "average_response_time_ms": round(sum(latencies) / len(latencies), 2),
        "risk_distribution": dict(risk_counts),
        "recent_predictions": list(prediction_history)[-25:],
    }


def _record_prediction(prediction: PredictionResponse, response_time_ms: float) -> None:
    prediction_history.append(
        {
            "customer_id": prediction.customer_id,
            "churn_probability": prediction.churn_probability,
            "predicted_class": prediction.predicted_class,
            "risk_level": prediction.risk_level,
            "model_version": prediction.model_version,
            "response_time_ms": round(response_time_ms, 2),
            "timestamp": prediction.timestamp.isoformat(),
        }
    )

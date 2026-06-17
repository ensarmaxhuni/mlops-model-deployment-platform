"""Application configuration helpers.

The project intentionally keeps configuration lightweight so it deploys cleanly
to local machines, Docker, Render, Railway, and Streamlit Community Cloud.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    project_name: str = os.getenv("PROJECT_NAME", "MLOps Model Deployment Platform")
    app_env: str = os.getenv("APP_ENV", "development")
    api_version: str = os.getenv("API_VERSION", "1.0.0")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    model_version: str = os.getenv("MODEL_VERSION", "1.0.0")
    author: str = os.getenv("MODEL_AUTHOR", "Ensar Maxhuni")
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")
    prediction_history_limit: int = _get_int("PREDICTION_HISTORY_LIMIT", 500)

    model_path: Path = PROJECT_ROOT / "model" / "model.pkl"
    metrics_path: Path = PROJECT_ROOT / "model" / "metrics.json"
    model_info_path: Path = PROJECT_ROOT / "model" / "model_info.json"
    sample_data_path: Path = PROJECT_ROOT / "data" / "sample_customers.csv"

    @property
    def cors_origin_list(self) -> list[str]:
        """Return normalized CORS origins."""

        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()


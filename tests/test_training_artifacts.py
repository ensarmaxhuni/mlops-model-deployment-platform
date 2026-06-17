"""Tests that validate generated model artifacts and training outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

from app.schemas import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture(scope="session", autouse=True)
def ensure_model_artifacts() -> None:
    if not (MODEL_DIR / "model.pkl").exists():
        subprocess.run([sys.executable, "model/train_model.py"], cwd=PROJECT_ROOT, check=True)


def test_model_artifacts_exist() -> None:
    assert (MODEL_DIR / "model.pkl").exists()
    assert (MODEL_DIR / "metrics.json").exists()
    assert (MODEL_DIR / "model_info.json").exists()
    assert (DATA_DIR / "sample_customers.csv").exists()


def test_metrics_artifact_schema() -> None:
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))

    assert set(metrics) == {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
    assert all(0 <= value <= 1 for value in metrics.values())


def test_model_info_artifact_schema() -> None:
    model_info = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))

    assert model_info["version"]
    assert model_info["algorithm"] == "GradientBoostingClassifier"
    assert model_info["features"] == FEATURE_COLUMNS
    assert 0 < model_info["decision_threshold"] < 1
    assert model_info["dataset_size"] > 0
    assert model_info["training_rows"] > 0
    assert model_info["test_rows"] > 0
    assert model_info["metrics"] == json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))


def test_sample_customer_csv_contract() -> None:
    sample = pd.read_csv(DATA_DIR / "sample_customers.csv")

    assert "customer_id" in sample.columns
    assert list(sample.columns[1:]) == FEATURE_COLUMNS
    assert len(sample) >= 10


def test_saved_model_can_score_sample_customer() -> None:
    model = joblib.load(MODEL_DIR / "model.pkl")
    sample = pd.read_csv(DATA_DIR / "sample_customers.csv")
    probabilities = model.predict_proba(sample[FEATURE_COLUMNS].head(3))[:, 1]

    assert len(probabilities) == 3
    assert all(0 <= probability <= 1 for probability in probabilities)

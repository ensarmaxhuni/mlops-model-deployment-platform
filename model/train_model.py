"""Train and version the customer churn model.

Run from the project root:

    python model/train_model.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"
SAMPLE_DATA_PATH = DATA_DIR / "sample_customers.csv"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "usage_frequency",
    "satisfaction_score",
    "number_of_services",
]
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "product_plan"]
FEATURES = [
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


def generate_customer_churn_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic customer churn dataset."""

    rng = np.random.default_rng(random_state)

    product_plan = rng.choice(
        ["Basic", "Standard", "Premium", "Enterprise"],
        size=n_samples,
        p=[0.34, 0.38, 0.20, 0.08],
    )
    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.52, 0.31, 0.17],
    )
    payment_method = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        size=n_samples,
        p=[0.38, 0.29, 0.24, 0.09],
    )

    plan_base_charge = {
        "Basic": 42,
        "Standard": 72,
        "Premium": 108,
        "Enterprise": 162,
    }
    monthly_charges = np.array([plan_base_charge[plan] for plan in product_plan], dtype=float)
    monthly_charges += rng.normal(0, 13, n_samples)
    monthly_charges = np.clip(monthly_charges, 20, 240).round(2)

    tenure_months = rng.integers(1, 73, size=n_samples)
    total_charges = (monthly_charges * tenure_months + rng.normal(0, 120, n_samples)).clip(0).round(2)

    support_tickets = rng.poisson(lam=1.7, size=n_samples)
    support_tickets += (contract_type == "Month-to-month").astype(int) * rng.binomial(2, 0.28, n_samples)
    support_tickets = np.clip(support_tickets, 0, 12)

    usage_frequency = rng.normal(loc=17, scale=7, size=n_samples)
    usage_frequency -= (support_tickets * 0.8)
    usage_frequency += np.where(product_plan == "Enterprise", 5, 0)
    usage_frequency = np.clip(np.rint(usage_frequency), 0, 45).astype(int)

    satisfaction_score = rng.normal(loc=7.2, scale=1.65, size=n_samples)
    satisfaction_score -= support_tickets * 0.28
    satisfaction_score += np.where(contract_type == "Two year", 0.35, 0)
    satisfaction_score = np.clip(satisfaction_score, 1, 10).round(1)

    services_by_plan = {
        "Basic": (1, 3),
        "Standard": (2, 5),
        "Premium": (4, 8),
        "Enterprise": (6, 12),
    }
    number_of_services = np.array(
        [rng.integers(services_by_plan[plan][0], services_by_plan[plan][1] + 1) for plan in product_plan]
    )

    logit = 0.10
    logit += np.where(contract_type == "Month-to-month", 1.05, 0)
    logit += np.where(contract_type == "One year", 0.10, -0.45)
    logit += np.where(payment_method == "Electronic check", 0.42, 0)
    logit += np.where(payment_method == "Credit card", -0.18, 0)
    logit += np.where(product_plan == "Basic", 0.18, 0)
    logit += np.where(product_plan == "Enterprise", -0.24, 0)
    logit += support_tickets * 0.28
    logit += (6.8 - satisfaction_score) * 0.45
    logit += (monthly_charges - 75) * 0.007
    logit -= tenure_months * 0.036
    logit -= usage_frequency * 0.040
    logit -= number_of_services * 0.12
    logit += rng.normal(0, 0.28, n_samples)

    churn_probability = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "support_tickets": support_tickets,
            "usage_frequency": usage_frequency,
            "satisfaction_score": satisfaction_score,
            "product_plan": product_plan,
            "number_of_services": number_of_services,
            "churned": churned,
        }
    )


def build_pipeline() -> Pipeline:
    """Build the preprocessing and classification pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    classifier = GradientBoostingClassifier(
        n_estimators=160,
        learning_rate=0.045,
        max_depth=3,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def find_optimal_threshold(probabilities: np.ndarray, y_true: pd.Series) -> float:
    """Choose a decision threshold that maximizes F1 on the holdout set."""

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(np.nanargmax(f1_scores))
    return round(float(thresholds[best_index]), 4)


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Calculate model performance metrics."""

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    """Train the churn model and persist all deployment artifacts."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = generate_customer_churn_data()
    x = dataset[FEATURES]
    y = dataset["churned"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    holdout_probabilities = model.predict_proba(x_test)[:, 1]
    decision_threshold = find_optimal_threshold(holdout_probabilities, y_test)
    metrics = evaluate_model(model, x_test, y_test, decision_threshold)

    joblib.dump(model, MODEL_PATH)
    write_json(METRICS_PATH, metrics)

    training_date = datetime.now(timezone.utc).isoformat()
    model_info = {
        "model_name": "Customer Churn Gradient Boosting Classifier",
        "version": os.getenv("MODEL_VERSION", "1.0.0"),
        "algorithm": "GradientBoostingClassifier",
        "training_date": training_date,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": "churned",
        "decision_threshold": decision_threshold,
        "dataset_size": int(len(dataset)),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "positive_class_rate": round(float(y.mean()), 4),
        "metrics": metrics,
        "author": os.getenv("MODEL_AUTHOR", "Ensar Maxhuni"),
    }
    write_json(MODEL_INFO_PATH, model_info)

    sample = dataset[FEATURES].head(50).copy()
    sample.insert(0, "customer_id", [f"CUST-{10000 + idx}" for idx in range(len(sample))])
    sample.to_csv(SAMPLE_DATA_PATH, index=False)

    print(f"Model artifact saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Model info saved to: {MODEL_INFO_PATH}")
    print(f"Sample customer CSV saved to: {SAMPLE_DATA_PATH}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()

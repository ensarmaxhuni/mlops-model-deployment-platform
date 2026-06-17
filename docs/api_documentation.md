# API Documentation

Base URL for local development:

```text
http://localhost:8000
```

Interactive OpenAPI docs:

```text
http://localhost:8000/docs
```

## GET /

Returns platform status and documentation links.

Example response:

```json
{
  "service": "MLOps Model Deployment Platform",
  "status": "running",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

## GET /health

Returns service health and model artifact availability.

Example response:

```json
{
  "status": "healthy",
  "model_available": true,
  "model_loadable": true,
  "environment": "development",
  "detail": "Model artifacts are present and loadable.",
  "timestamp": "2026-06-15T22:30:00.000000Z"
}
```

## GET /model-info

Returns model metadata.

Key fields:

- `model_name`
- `version`
- `algorithm`
- `training_date`
- `features`
- `decision_threshold`
- `dataset_size`
- `metrics`
- `author`

## GET /metrics

Returns holdout model performance metrics.

Example response:

```json
{
  "accuracy": 0.82,
  "precision": 0.71,
  "recall": 0.64,
  "f1_score": 0.67,
  "roc_auc": 0.86
}
```

## GET /monitoring

Returns recent in-memory prediction monitoring.

Example response:

```json
{
  "total_predictions": 24,
  "average_churn_probability": 0.4312,
  "high_risk_customers": 5,
  "average_response_time_ms": 9.35,
  "risk_distribution": {
    "Low Risk": 10,
    "Medium Risk": 9,
    "High Risk": 5
  },
  "recent_predictions": []
}
```

## POST /predict

Scores one customer.

Request schema:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `customer_id` | string | no | Optional trace identifier |
| `tenure_months` | integer | yes | 0 to 120 |
| `monthly_charges` | number | yes | 0 to 500 |
| `total_charges` | number | yes | 0 to 50000 |
| `contract_type` | string | yes | Month-to-month, One year, Two year |
| `payment_method` | string | yes | Electronic check, Credit card, Bank transfer, Mailed check |
| `support_tickets` | integer | yes | 0 to 50 |
| `usage_frequency` | integer | yes | Product sessions in last 30 days |
| `satisfaction_score` | number | yes | 1 to 10 |
| `product_plan` | string | yes | Basic, Standard, Premium, Enterprise |
| `number_of_services` | integer | yes | 1 to 20 |

Example request:

```json
{
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
  "number_of_services": 3
}
```

Example response:

```json
{
  "customer_id": "CUST-10291",
  "churn_probability": 0.7342,
  "predicted_class": 1,
  "risk_level": "High Risk",
  "recommendation": "Prioritize immediate retention action, account review, and incentive offer.",
  "suggested_retention_action": "Escalate to retention team, review open issues, and prepare a targeted incentive.",
  "model_version": "1.0.0",
  "timestamp": "2026-06-15T22:30:00.000000Z"
}
```

## POST /batch-predict

Scores many customers.

Example request:

```json
{
  "customers": [
    {
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
      "number_of_services": 3
    }
  ]
}
```

Example response:

```json
{
  "predictions": [
    {
      "customer_id": "CUST-10291",
      "churn_probability": 0.7342,
      "predicted_class": 1,
      "risk_level": "High Risk",
      "recommendation": "Prioritize immediate retention action, account review, and incentive offer.",
      "suggested_retention_action": "Escalate to retention team, review open issues, and prepare a targeted incentive.",
      "model_version": "1.0.0",
      "timestamp": "2026-06-15T22:30:00.000000Z"
    }
  ],
  "total_customers": 1,
  "model_version": "1.0.0",
  "timestamp": "2026-06-15T22:30:00.000000Z"
}
```

## Error Handling

Invalid payloads return HTTP `422` with Pydantic validation details.

Missing model artifacts return HTTP `503` with an action message telling the operator to run:

```bash
python model/train_model.py
```

# Architecture

## System Overview

The MLOps Model Deployment Platform is split into four layers:

1. Data and training layer
2. Model artifact layer
3. FastAPI serving layer
4. Streamlit operations dashboard

```text
data/sample_customers.csv
        |
        v
model/train_model.py
        |
        +--> model/model.pkl
        +--> model/metrics.json
        +--> model/model_info.json
        |
        v
app/main.py FastAPI service
        |
        +--> /predict
        +--> /batch-predict
        +--> /model-info
        +--> /metrics
        +--> /monitoring
        |
        v
dashboard/streamlit_app.py
```

## Data Flow

```text
Synthetic customer generator
        |
        v
Feature matrix and churn labels
        |
        v
Train/test split
        |
        v
Preprocessing pipeline
        |
        v
Gradient Boosting classifier
        |
        v
Persisted model and metadata artifacts
```

The model uses a mix of numerical and categorical business features:

- Tenure months
- Monthly charges
- Total charges
- Contract type
- Payment method
- Support tickets
- Usage frequency
- Satisfaction score
- Product plan
- Number of services

## Model Flow

The training script builds a Scikit-learn `Pipeline`:

```text
Raw customer features
        |
        +--> Numeric features -> StandardScaler
        |
        +--> Categorical features -> OneHotEncoder
        |
        v
GradientBoostingClassifier
        |
        v
Churn probability
```

The feature list is stored in `model/model_info.json` and validated by the prediction service before predictions are served. This protects the API from using an artifact with a different feature contract.

The model metadata also stores a holdout-optimized decision threshold. The API uses this threshold for `predicted_class` while still returning the raw churn probability and fixed business risk bands.

## API Flow

```text
Client request
        |
        v
Pydantic schema validation
        |
        v
Prediction service loads model artifacts
        |
        v
Feature frame assembled in trained order
        |
        v
Model probability generated
        |
        v
Optimized decision threshold applied
        |
        v
Risk level and retention action mapped
        |
        v
Typed JSON response returned
```

Risk thresholds:

- 0.00 to 0.35: Low Risk
- 0.36 to 0.70: Medium Risk
- 0.71 to 1.00: High Risk

## Monitoring Simulation

The API stores recent prediction summaries in memory:

- Total predictions
- Average churn probability
- High-risk customer count
- Average response time
- Risk distribution
- Recent prediction stream

The Streamlit dashboard also stores session-level history so the demo remains useful if the API monitoring endpoint is unavailable.

## Reliability Checks

The `/health` endpoint verifies that model artifacts are present and loadable. Docker Compose uses service health checks so the dashboard waits for a healthy API before starting.

## Deployment Diagram

```text
Developer machine or cloud host
        |
        +-- Container: api
        |       |
        |       +-- Uvicorn
        |       +-- FastAPI
        |       +-- Scikit-learn model artifact
        |
        +-- Container: dashboard
                |
                +-- Streamlit
                +-- Requests to http://api:8000
```

For single-service deployment platforms, deploy the FastAPI service and Streamlit dashboard as two separate services that share the same repository and model artifacts.

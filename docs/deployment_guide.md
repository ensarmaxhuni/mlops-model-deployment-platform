# Deployment Guide

## Local Deployment

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model artifacts:

```bash
python model/train_model.py
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Start the dashboard in another terminal:

```bash
streamlit run dashboard/streamlit_app.py
```

Local URLs:

- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

## Docker Deployment

Build and run both services:

```bash
docker-compose up --build
```

Docker Compose starts:

- `api` on port `8000`
- `dashboard` on port `8501`

The dashboard uses `API_URL=http://api:8000` inside the Docker network.
The API and dashboard services include health checks so Compose can surface unhealthy containers quickly.

## GitHub Actions

The repository includes `.github/workflows/ci.yml`.

The workflow:

- Installs Python 3.11
- Installs dependencies from `requirements.txt`
- Trains model artifacts with `python model/train_model.py`
- Runs the Pytest suite

## Render Deployment Notes

Deploy the FastAPI backend as a Web Service:

- Build command: `pip install -r requirements.txt && python model/train_model.py`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Runtime: Python 3.11+

Recommended environment variables:

```text
APP_ENV=production
LOG_LEVEL=INFO
MODEL_VERSION=1.0.0
CORS_ORIGINS=*
```

Deploy the Streamlit dashboard as a second Web Service:

- Build command: `pip install -r requirements.txt && python model/train_model.py`
- Start command: `streamlit run dashboard/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
- Environment variable: `API_URL=https://your-render-api-url`

## Railway Deployment Notes

Use two Railway services from the same GitHub repository:

FastAPI service:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Streamlit service:

```bash
streamlit run dashboard/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

Set `API_URL` on the dashboard service to the deployed API URL.

## Streamlit Community Cloud Notes

Streamlit Community Cloud can host the dashboard. The FastAPI backend should be deployed separately on Render, Railway, or another API host.

Dashboard settings:

- Main file path: `dashboard/streamlit_app.py`
- Python version: 3.11+
- Secrets or environment variable: `API_URL=https://your-api-host`

## Production Considerations

Before using this architecture with real customer data, add:

- Authentication and authorization
- Persistent prediction storage
- Model registry or MLflow tracking server
- Feature store or validated batch feature pipeline
- Data drift and concept drift monitoring
- CI/CD pipeline for tests and Docker builds
- Secrets management
- Rate limits and audit logging

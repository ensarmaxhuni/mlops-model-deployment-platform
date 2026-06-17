"""Streamlit dashboard for model operations and business churn insights."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_INFO_PATH = PROJECT_ROOT / "model" / "model_info.json"
METRICS_PATH = PROJECT_ROOT / "model" / "metrics.json"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_customers.csv"
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

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

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = ["Electronic check", "Credit card", "Bank transfer", "Mailed check"]
PRODUCT_PLANS = ["Basic", "Standard", "Premium", "Enterprise"]


st.set_page_config(
    page_title="MLOps Model Deployment Platform",
    page_icon="ML",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: #111111;
                color: #f4f1e8;
            }
            [data-testid="stSidebar"] {
                background: #181818;
                border-right: 1px solid #2d2a24;
            }
            h1, h2, h3 {
                color: #f8f4e8;
                letter-spacing: 0;
            }
            .platform-header {
                border: 1px solid #31302c;
                border-radius: 8px;
                padding: 20px 22px;
                background: linear-gradient(135deg, #1b1b1b 0%, #24211b 100%);
                margin-bottom: 16px;
            }
            .platform-header h1 {
                margin: 0 0 6px 0;
                font-size: 2.15rem;
            }
            .platform-header p {
                margin: 0;
                color: #c9c2b2;
                font-size: 1rem;
            }
            .metric-card {
                border: 1px solid #31302c;
                border-radius: 8px;
                padding: 16px;
                background: #1a1a1a;
                min-height: 112px;
            }
            .metric-label {
                color: #b8b0a0;
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 8px;
            }
            .metric-value {
                color: #f8f4e8;
                font-size: 1.9rem;
                font-weight: 700;
                line-height: 1.1;
            }
            .metric-note {
                color: #a59d8e;
                font-size: 0.82rem;
                margin-top: 8px;
            }
            .risk-low {
                color: #2dd4bf;
                font-weight: 700;
            }
            .risk-medium {
                color: #f5c542;
                font-weight: 700;
            }
            .risk-high {
                color: #ff6b6b;
                font-weight: 700;
            }
            div[data-testid="stDataFrame"] {
                border: 1px solid #31302c;
            }
            .stButton > button, .stDownloadButton > button {
                border-radius: 6px;
                border: 1px solid #39bfa7;
                background: #1f6f64;
                color: #ffffff;
                font-weight: 650;
            }
            .stButton > button:hover, .stDownloadButton > button:hover {
                border-color: #55dac1;
                background: #288878;
                color: #ffffff;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 6px;
                background: #1c1c1c;
                border: 1px solid #31302c;
                color: #e8dfcf;
            }
            .stTabs [aria-selected="true"] {
                background: #1f6f64;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def api_get(endpoint: str, timeout: int = 5) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def api_post(endpoint: str, payload: dict[str, Any], timeout: int = 20) -> tuple[dict[str, Any] | None, str | None, float]:
    started = perf_counter()
    try:
        response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json(), None, (perf_counter() - started) * 1000
    except requests.RequestException as exc:
        return None, str(exc), (perf_counter() - started) * 1000


def load_local_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_model_info() -> dict[str, Any]:
    payload, _ = api_get("/model-info")
    return payload or load_local_json(MODEL_INFO_PATH)


def get_metrics() -> dict[str, Any]:
    payload, _ = api_get("/metrics")
    return payload or load_local_json(METRICS_PATH)


def get_monitoring() -> dict[str, Any]:
    payload, _ = api_get("/monitoring")
    if payload:
        return payload
    history = st.session_state.prediction_history
    if not history:
        return {
            "total_predictions": 0,
            "average_churn_probability": 0,
            "high_risk_customers": 0,
            "average_response_time_ms": 0,
            "risk_distribution": {},
            "recent_predictions": [],
        }
    probabilities = [item["churn_probability"] for item in history]
    latencies = [item.get("response_time_ms", 0) for item in history]
    risk_counts = pd.Series([item["risk_level"] for item in history]).value_counts().to_dict()
    return {
        "total_predictions": len(history),
        "average_churn_probability": round(sum(probabilities) / len(probabilities), 4),
        "high_risk_customers": risk_counts.get("High Risk", 0),
        "average_response_time_ms": round(sum(latencies) / len(latencies), 2),
        "risk_distribution": risk_counts,
        "recent_predictions": history[-25:],
    }


def record_prediction(prediction: dict[str, Any], response_time_ms: float) -> None:
    st.session_state.prediction_history.append(
        {
            "customer_id": prediction.get("customer_id"),
            "churn_probability": prediction.get("churn_probability", 0),
            "predicted_class": prediction.get("predicted_class"),
            "risk_level": prediction.get("risk_level"),
            "model_version": prediction.get("model_version"),
            "response_time_ms": round(response_time_ms, 2),
            "timestamp": prediction.get("timestamp", datetime.now(timezone.utc).isoformat()),
        }
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="platform-header">
            <h1>MLOps Model Deployment Platform</h1>
            <p>Customer churn prediction, model serving, monitoring, and business retention intelligence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_risk_distribution(risk_distribution: dict[str, int]) -> go.Figure:
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    values = [risk_distribution.get(label, 0) for label in labels]
    fig = px.bar(
        x=labels,
        y=values,
        color=labels,
        color_discrete_map={
            "Low Risk": "#2dd4bf",
            "Medium Risk": "#f5c542",
            "High Risk": "#ff6b6b",
        },
        labels={"x": "Risk category", "y": "Predictions"},
        template="plotly_dark",
    )
    fig.update_layout(
        showlegend=False,
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#111111",
        plot_bgcolor="#181818",
    )
    return fig


def plot_probability_distribution(history: list[dict[str, Any]]) -> go.Figure:
    probabilities = [item.get("churn_probability", 0) for item in history]
    fig = px.histogram(
        x=probabilities,
        nbins=12,
        labels={"x": "Churn probability", "y": "Customers"},
        template="plotly_dark",
        color_discrete_sequence=["#39bfa7"],
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#111111",
        plot_bgcolor="#181818",
    )
    return fig


def plot_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"color": "#f8f4e8", "size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#f8f4e8"},
                "bar": {"color": "#39bfa7"},
                "bgcolor": "#181818",
                "borderwidth": 1,
                "bordercolor": "#31302c",
                "steps": [
                    {"range": [0, 35], "color": "#143f3a"},
                    {"range": [35, 70], "color": "#4d3f16"},
                    {"range": [70, 100], "color": "#4d2020"},
                ],
                "threshold": {
                    "line": {"color": "#ff6b6b", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#111111",
        font={"color": "#f8f4e8"},
    )
    return fig


def render_overview(model_info: dict[str, Any]) -> None:
    health, _ = api_get("/health")
    monitoring = get_monitoring()
    status = "Online" if health and health.get("status") == "healthy" else "Offline"
    model_version = model_info.get("version", "unknown")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Total Predictions", str(monitoring["total_predictions"]), "API and dashboard session")
    with col2:
        render_metric_card(
            "Avg Churn Probability",
            f"{monitoring['average_churn_probability'] * 100:.1f}%",
            "Rolling in-memory metric",
        )
    with col3:
        render_metric_card("High-Risk Customers", str(monitoring["high_risk_customers"]), "Retention priority queue")
    with col4:
        render_metric_card("Model Version", model_version, model_info.get("algorithm", "classifier"))
    with col5:
        render_metric_card("System Status", status, f"API: {API_URL}")

    st.subheader("Operational Monitoring")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(plot_risk_distribution(monitoring["risk_distribution"]), use_container_width=True)
    with chart_col2:
        st.plotly_chart(plot_probability_distribution(monitoring["recent_predictions"]), use_container_width=True)

    if monitoring["recent_predictions"]:
        st.subheader("Recent Prediction Stream")
        st.dataframe(pd.DataFrame(monitoring["recent_predictions"]), use_container_width=True, hide_index=True)
    else:
        st.info("No predictions have been recorded yet. Run a single or batch prediction to populate monitoring.")


def customer_form() -> dict[str, Any]:
    with st.form("single_prediction_form"):
        left, middle, right = st.columns(3)
        with left:
            customer_id = st.text_input("Customer ID", value="CUST-10291")
            tenure_months = st.number_input("Tenure Months", min_value=0, max_value=120, value=18)
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=84.5, step=1.0)
            total_charges = st.number_input("Total Charges", min_value=0.0, max_value=50000.0, value=1521.0, step=10.0)
        with middle:
            contract_type = st.selectbox("Contract Type", CONTRACT_TYPES)
            payment_method = st.selectbox("Payment Method", PAYMENT_METHODS)
            support_tickets = st.number_input("Support Tickets", min_value=0, max_value=50, value=4)
        with right:
            usage_frequency = st.number_input("Usage Frequency", min_value=0, max_value=100, value=9)
            satisfaction_score = st.slider("Satisfaction Score", min_value=1.0, max_value=10.0, value=5.8, step=0.1)
            product_plan = st.selectbox("Product Plan", PRODUCT_PLANS, index=1)
            number_of_services = st.number_input("Number of Services", min_value=1, max_value=20, value=3)

        submitted = st.form_submit_button("Generate Prediction", use_container_width=True)

    return {
        "submitted": submitted,
        "payload": {
            "customer_id": customer_id,
            "tenure_months": int(tenure_months),
            "monthly_charges": float(monthly_charges),
            "total_charges": float(total_charges),
            "contract_type": contract_type,
            "payment_method": payment_method,
            "support_tickets": int(support_tickets),
            "usage_frequency": int(usage_frequency),
            "satisfaction_score": float(satisfaction_score),
            "product_plan": product_plan,
            "number_of_services": int(number_of_services),
        },
    }


def render_single_prediction() -> None:
    st.subheader("Single Customer Prediction")
    form_result = customer_form()

    if form_result["submitted"]:
        result, error, response_time_ms = api_post("/predict", form_result["payload"])
        if error or not result:
            st.error(f"Prediction request failed: {error}")
            st.code(json.dumps(form_result["payload"], indent=2), language="json")
            return

        record_prediction(result, response_time_ms)
        probability = float(result["churn_probability"])
        risk_level = result["risk_level"]
        risk_class = {
            "Low Risk": "risk-low",
            "Medium Risk": "risk-medium",
            "High Risk": "risk-high",
        }.get(risk_level, "risk-medium")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(plot_gauge(probability), use_container_width=True)
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Prediction Result</div>
                    <div class="metric-value">{probability * 100:.1f}%</div>
                    <div class="{risk_class}">{risk_level}</div>
                    <div class="metric-note">Model version {result['model_version']} | {response_time_ms:.0f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(result["recommendation"])
            st.info(result["suggested_retention_action"])


def render_batch_prediction() -> None:
    st.subheader("Batch Prediction Tool")
    if SAMPLE_DATA_PATH.exists():
        sample_bytes = SAMPLE_DATA_PATH.read_bytes()
        st.download_button(
            "Download Sample Input CSV",
            data=sample_bytes,
            file_name="sample_customers.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is None:
        st.caption("Required columns: " + ", ".join(FEATURE_COLUMNS))
        return

    frame = pd.read_csv(uploaded)
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    customers = frame[[column for column in ["customer_id", *FEATURE_COLUMNS] if column in frame.columns]]
    customers = customers.where(pd.notnull(customers), None)
    payload = {"customers": customers.to_dict(orient="records")}
    result, error, response_time_ms = api_post("/batch-predict", payload, timeout=60)
    if error or not result:
        st.error(f"Batch prediction failed: {error}")
        return

    for prediction in result["predictions"]:
        record_prediction(prediction, response_time_ms / max(len(result["predictions"]), 1))

    prediction_frame = pd.DataFrame(result["predictions"])
    merged = pd.concat([frame.reset_index(drop=True), prediction_frame.reset_index(drop=True)], axis=1)
    st.success(f"Scored {len(merged)} customers in {response_time_ms:.0f} ms.")
    st.dataframe(merged, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Predictions CSV",
        data=merged.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_model_performance(metrics: dict[str, Any]) -> None:
    st.subheader("Model Performance")
    if not metrics:
        st.warning("Metrics are unavailable. Train the model first.")
        return

    cols = st.columns(5)
    metric_labels = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1_score"),
        ("ROC AUC", "roc_auc"),
    ]
    for column, (label, key) in zip(cols, metric_labels):
        with column:
            render_metric_card(label, f"{metrics.get(key, 0) * 100:.1f}%", "Holdout validation")

    metric_frame = pd.DataFrame(
        {"metric": [label for label, _ in metric_labels], "score": [metrics.get(key, 0) for _, key in metric_labels]}
    )
    fig = px.bar(
        metric_frame,
        x="metric",
        y="score",
        text="score",
        color="metric",
        color_discrete_sequence=["#39bfa7", "#f5c542", "#ff6b6b", "#8bd3dd", "#f2a65a"],
        template="plotly_dark",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        showlegend=False,
        yaxis_range=[0, 1],
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#111111",
        plot_bgcolor="#181818",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_information(model_info: dict[str, Any]) -> None:
    st.subheader("Model Information")
    if not model_info:
        st.warning("Model metadata is unavailable. Train the model first.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Version", str(model_info.get("version", "unknown")), model_info.get("model_name", ""))
    with col2:
        render_metric_card("Algorithm", model_info.get("algorithm", "unknown"), "Production classifier")
    with col3:
        render_metric_card("Dataset Size", f"{model_info.get('dataset_size', 0):,}", "Synthetic training records")
    with col4:
        render_metric_card("Positive Class Rate", f"{model_info.get('positive_class_rate', 0) * 100:.1f}%", "Churn prevalence")

    details = {
        "Training date": model_info.get("training_date"),
        "Author": model_info.get("author"),
        "Target": model_info.get("target"),
        "Decision threshold": model_info.get("decision_threshold"),
        "Training rows": model_info.get("training_rows"),
        "Test rows": model_info.get("test_rows"),
    }
    st.dataframe(pd.DataFrame(details.items(), columns=["Attribute", "Value"]), use_container_width=True, hide_index=True)
    st.write("Feature contract")
    st.dataframe(pd.DataFrame({"feature": model_info.get("features", [])}), use_container_width=True, hide_index=True)


def render_api_playground() -> None:
    st.subheader("API Playground")
    example_payload = {
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
    st.code(json.dumps(example_payload, indent=2), language="json")
    endpoint_frame = pd.DataFrame(
        [
            {"Method": "GET", "Endpoint": "/", "Purpose": "Platform status"},
            {"Method": "GET", "Endpoint": "/health", "Purpose": "Service health check"},
            {"Method": "GET", "Endpoint": "/model-info", "Purpose": "Model metadata and feature contract"},
            {"Method": "GET", "Endpoint": "/metrics", "Purpose": "Model performance metrics"},
            {"Method": "GET", "Endpoint": "/monitoring", "Purpose": "Prediction monitoring summary"},
            {"Method": "POST", "Endpoint": "/predict", "Purpose": "Single customer churn prediction"},
            {"Method": "POST", "Endpoint": "/batch-predict", "Purpose": "Batch customer scoring"},
        ]
    )
    st.dataframe(endpoint_frame, use_container_width=True, hide_index=True)
    st.link_button("Open FastAPI Docs", f"{API_URL}/docs", use_container_width=True)


def main() -> None:
    inject_css()
    init_state()
    render_header()

    model_info = get_model_info()
    metrics = get_metrics()

    with st.sidebar:
        st.title("Operations")
        st.caption(f"API base URL: {API_URL}")
        selected = st.radio(
            "Navigation",
            [
                "Executive Overview",
                "Single Prediction",
                "Batch Prediction",
                "Model Performance",
                "Model Information",
                "API Playground",
            ],
        )
        st.divider()
        health, error = api_get("/health")
        if health and health.get("status") == "healthy":
            st.success("API online")
        else:
            st.error("API offline")
            if error:
                st.caption(error)

    if selected == "Executive Overview":
        render_overview(model_info)
    elif selected == "Single Prediction":
        render_single_prediction()
    elif selected == "Batch Prediction":
        render_batch_prediction()
    elif selected == "Model Performance":
        render_model_performance(metrics)
    elif selected == "Model Information":
        render_model_information(model_info)
    else:
        render_api_playground()


if __name__ == "__main__":
    main()

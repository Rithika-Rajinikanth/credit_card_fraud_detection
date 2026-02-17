from fastapi import FastAPI, Request
import pandas as pd
import joblib
import logging
import time

from prometheus_client import Counter, Histogram, generate_latest
from src.schemas import Transaction
from src.logger import generate_request_id

# --------------------------------------------------
# App initialization
# --------------------------------------------------
app = FastAPI(title="Fraud Detection Service")

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
model = joblib.load("artifacts/model.pkl")
threshold = float(joblib.load("artifacts/threshold.pkl"))

# --------------------------------------------------
# Feature list (training–serving parity)
# --------------------------------------------------
FEATURES = [
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28",
    "Amount","Hour"
]

# --------------------------------------------------
# Prometheus Metrics
# --------------------------------------------------
REQUEST_COUNT = Counter(
    "fraud_requests_total",
    "Total fraud scoring requests"
)

FRAUD_BLOCKED_COUNT = Counter(
    "fraud_blocked_total",
    "Total blocked transactions"
)

LATENCY = Histogram(
    "fraud_latency_seconds",
    "Fraud scoring latency"
)

# --------------------------------------------------
# Rule Engine
# --------------------------------------------------
def rule_override(txn: dict) -> bool:
    """
    Deterministic business rules.
    Returns True if transaction must be blocked.
    """
    if txn["Amount"] < 50 and txn.get("V14", 0) < -10:
        return True
    return False

# --------------------------------------------------
# Scoring Endpoint
# --------------------------------------------------
@app.post("/score")
def score_transaction(txn: Transaction, request: Request):

    request_id = generate_request_id()
    REQUEST_COUNT.inc()

    start_time = time.time()
    logger.info(f"[{request_id}] Incoming transaction")

    # Ensure feature completeness & order
    txn_dict = txn.dict()
    data = {f: float(txn_dict.get(f, 0)) for f in FEATURES}
    df = pd.DataFrame([data], columns=FEATURES)

    # ML scoring
    prob = float(model.predict_proba(df)[0, 1])
    ml_decision = "BLOCK" if prob >= threshold else "APPROVE"

    # Rule override
    rule_block = rule_override(data)

    if rule_block:
        final_decision = "BLOCK"
        decision_source = "RULE_OVERRIDE"
        FRAUD_BLOCKED_COUNT.inc()
    else:
        final_decision = ml_decision
        decision_source = "ML_MODEL"
        if final_decision == "BLOCK":
            FRAUD_BLOCKED_COUNT.inc()

    LATENCY.observe(time.time() - start_time)

    logger.info(
        f"[{request_id}] prob={prob:.4f} "
        f"decision={final_decision} source={decision_source}"
    )

    return {
        "request_id": request_id,
        "fraud_probability": round(prob, 4),
        "decision": final_decision,
        "decision_source": decision_source
    }

# --------------------------------------------------
# Prometheus Metrics Endpoint
# --------------------------------------------------
@app.get("/metrics")
def metrics():
    return generate_latest()

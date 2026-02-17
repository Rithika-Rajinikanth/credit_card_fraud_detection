from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "fraud_requests_total",
    "Total fraud scoring requests"
)

FRAUD_COUNT = Counter(
    "fraud_blocked_total",
    "Total blocked frauds"
)

LATENCY = Histogram(
    "fraud_latency_seconds",
    "Scoring latency"
)

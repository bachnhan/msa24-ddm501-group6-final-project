from prometheus_client import Counter, Histogram, Gauge

# --- 1. HTTP Metrics (Used by Middleware) ---
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total number of HTTP requests', 
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 
    'HTTP request latency in seconds', 
    ['method', 'endpoint']
)

# --- 2. Model Lifecycle Metrics (Used by app.model) ---
MODEL_LOADED = Gauge(
    'model_loaded', 
    'Current model load status (1=OK, 0=Fail)'
)
MODEL_LAST_RELOAD = Gauge(
    'model_last_reload_seconds', 
    'Timestamp of last model reload'
)
MODEL_INFO = Gauge(
    'model_info', 
    'Model metadata', 
    ['version', 'type']
)

# --- 3. Prediction Performance Metrics ---
PREDICTION_LATENCY = Histogram(
    'prediction_latency_ms', 
    'Time taken for churn prediction in milliseconds'
)
PREDICTION_TOTAL = Counter(
    'prediction_total', 
    'Total number of churn predictions made'
)
PREDICTION_CHURN_TOTAL = Counter(
    'prediction_churn_total', 
    'Total number of churn events predicted'
)
MODEL_ERROR_TOTAL = Counter(
    'model_error_total', 
    'Total number of prediction errors'
)

# --- 4. Responsible AI & Bias Monitoring (Rubric 3.1.5) ---
PREDICTION_BY_GENDER = Histogram(
    'prediction_by_gender', 
    'Churn probability distribution by gender', 
    ['model_version', 'gender']
)

# --- 5. Helper Functions ---
def count_implemented_metrics():
    """Returns the number of active metrics for the homepage info."""
    # List of all defined metric objects
    metrics = [
        REQUEST_COUNT, REQUEST_LATENCY, MODEL_LOADED, 
        MODEL_LAST_RELOAD, MODEL_INFO, PREDICTION_LATENCY, 
        PREDICTION_TOTAL, PREDICTION_CHURN_TOTAL, 
        MODEL_ERROR_TOTAL, PREDICTION_BY_GENDER
    ]
    implemented = len([m for m in metrics if m is not None])
    return implemented, len(metrics)

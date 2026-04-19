from prometheus_client import Counter, Histogram, Gauge, Info

# --- 1. HTTP Metrics ---
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

# --- 2. Model Lifecycle Metrics ---
MODEL_LOADED = Gauge(
    'ml_model_loaded', 
    'Current model load status (1=OK, 0=Fail)'
)
MODEL_LAST_RELOAD = Gauge(
    'ml_model_last_reload_seconds', 
    'Timestamp of last model reload'
)
MODEL_INFO = Info(
    'ml_model_info', 
    'Model metadata'
)

# --- 3. Prediction Activity Metrics ---
PREDICTION_COUNT = Counter(
    'ml_predictions_total', 
    'Total number of churn predictions made',
    labelnames=['model_version']
)
PREDICTION_LATENCY = Histogram(
    'ml_prediction_duration_seconds', 
    'Time taken for churn prediction in seconds'
)
PREDICTION_VALUE = Histogram(
    'ml_prediction_value', 
    'Distribution of prediction values'
)
PREDICTION_ERRORS = Counter(
    'ml_prediction_errors_total', 
    'Total number of prediction errors',
    labelnames=['model_version']
)

# --- 4. Responsible AI / Gender Metrics (Explicitly defined for app/main.py) ---
PREDICTION_BY_GENDER = Histogram(
    'prediction_by_gender', 
    'Churn probability distribution by gender', 
    ['model_version', 'gender']
)

# --- 6. Aliases for Compatibility ---
PREDICTION_TOTAL = PREDICTION_COUNT
MODEL_ERROR_TOTAL = PREDICTION_ERRORS

# --- 6. Helper Functions ---
def get_all_metrics():
    """Returns a dictionary of all defined metrics for tests."""
    return {
        'REQUEST_COUNT': REQUEST_COUNT,
        'REQUEST_LATENCY': REQUEST_LATENCY,
        'PREDICTION_COUNT': PREDICTION_COUNT,
        'PREDICTION_LATENCY': PREDICTION_LATENCY,
        'PREDICTION_VALUE': PREDICTION_VALUE,
        'PREDICTION_ERRORS': PREDICTION_ERRORS,
        'MODEL_LOADED': MODEL_LOADED,
        'MODEL_INFO': MODEL_INFO,
        'MODEL_LAST_RELOAD': MODEL_LAST_RELOAD,
        'PREDICTION_BY_GENDER': PREDICTION_BY_GENDER
    }

def count_implemented_metrics():
    """Returns the number of active metrics."""
    metrics = get_all_metrics()
    implemented = len([m for m in metrics.values() if m is not None])
    return implemented, len(metrics)

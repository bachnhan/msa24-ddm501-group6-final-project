"""
Prometheus metrics definitions for the Customer Churn API.

This module defines all Prometheus metrics used for monitoring the application.



Metrics Types:
- Counter: Cumulative values that only increase (e.g., total requests)
- Gauge: Values that can go up or down (e.g., current temperature)
- Histogram: Distribution of values in buckets (e.g., request latency)
- Summary: Similar to histogram with quantiles
- Info: Key-value pairs for static information

Run the API and check metrics at: http://localhost:8000/metrics
"""

from prometheus_client import Counter, Histogram, Gauge, Info

# =============================================================================
# Application Metrics (HTTP Requests)
# =============================================================================


REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status']
)


REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)


# =============================================================================
# ML-Specific Metrics
# =============================================================================


PREDICTION_COUNT = Counter(
    'ml_predictions_total',
    'Total number of predictions made',
    ['model_version']
)



PREDICTION_LATENCY = Histogram(
    'ml_prediction_duration_seconds',
    'Time to generate a prediction',
    ['model_version'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)



PREDICTION_VALUE = Histogram(
    'ml_prediction_value',
    'Distribution of prediction values',
    ['model_version'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# RESPONSIBLE AI: Fairness Monitoring
# -----------------------------------------------------------------------------
PREDICTION_BY_GENDER = Histogram(
    'ml_prediction_by_gender',
    'Distribution of prediction values by gender group',
    ['model_version', 'gender'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
# -----------------------------------------------------------------------------



PREDICTION_ERRORS = Counter(
    'ml_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type', 'model_version']
)


# =============================================================================
# Model Status Metrics
# =============================================================================


MODEL_LOADED = Gauge(
    'ml_model_loaded',
    'Whether the ML model is loaded (1) or not (0)'
)



MODEL_INFO = Info(
    'ml_model',
    'Information about the loaded ML model'
)



MODEL_LAST_RELOAD = Gauge(
    'ml_model_last_reload_timestamp',
    'Unix timestamp of last model reload'
)


# =============================================================================
# Batch Prediction Metrics (BONUS)
# =============================================================================


BATCH_SIZE = Histogram(
    'ml_batch_prediction_size',
    'Size of batch prediction requests',
    buckets=[1, 5, 10, 25, 50, 100]
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_all_metrics():
    """Return a dictionary of all defined metrics for inspection."""
    return {
        'request_count': REQUEST_COUNT,
        'request_latency': REQUEST_LATENCY,
        'prediction_count': PREDICTION_COUNT,
        'prediction_latency': PREDICTION_LATENCY,
        'prediction_value': PREDICTION_VALUE,
        'prediction_errors': PREDICTION_ERRORS,
        'model_loaded': MODEL_LOADED,
        'model_info': MODEL_INFO,
        'model_last_reload': MODEL_LAST_RELOAD,
        'batch_size': BATCH_SIZE,
    }


def count_implemented_metrics():
    """Count how many metrics have been implemented."""
    metrics = get_all_metrics()
    implemented = sum(1 for m in metrics.values() if m is not None)
    return implemented, len(metrics)

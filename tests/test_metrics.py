"""
Tests for metrics implementation.

These tests verify that metrics are properly defined and working.
"""

import pytest
from prometheus_client import REGISTRY

from app.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    PREDICTION_VALUE,
    PREDICTION_ERRORS,
    MODEL_LOADED,
    MODEL_INFO,
    MODEL_LAST_RELOAD,
    count_implemented_metrics,
    get_all_metrics,
)


class TestMetricDefinitions:
    """Test that metrics are properly defined."""
    
    def test_request_count_defined(self):
        """Test REQUEST_COUNT is defined."""
        assert REQUEST_COUNT is not None
        # Check it's a Counter
        assert hasattr(REQUEST_COUNT, 'inc')
    
    def test_request_latency_defined(self):
        """Test REQUEST_LATENCY is defined."""
        # This test will fail until students implement the metric
        if REQUEST_LATENCY is None:
            pytest.skip("REQUEST_LATENCY not implemented yet")
        assert hasattr(REQUEST_LATENCY, 'observe')
    
    def test_prediction_count_defined(self):
        """Test PREDICTION_COUNT is defined."""
        if PREDICTION_COUNT is None:
            pytest.skip("PREDICTION_COUNT not implemented yet")
        assert hasattr(PREDICTION_COUNT, 'inc')
    
    def test_prediction_latency_defined(self):
        """Test PREDICTION_LATENCY is defined."""
        if PREDICTION_LATENCY is None:
            pytest.skip("PREDICTION_LATENCY not implemented yet")
        assert hasattr(PREDICTION_LATENCY, 'observe')
    
    def test_prediction_value_defined(self):
        """Test PREDICTION_VALUE is defined."""
        if PREDICTION_VALUE is None:
            pytest.skip("PREDICTION_VALUE not implemented yet")
        assert hasattr(PREDICTION_VALUE, 'observe')
    
    def test_model_loaded_defined(self):
        """Test MODEL_LOADED is defined."""
        if MODEL_LOADED is None:
            pytest.skip("MODEL_LOADED not implemented yet")
        assert hasattr(MODEL_LOADED, 'set')
    
    def test_model_info_defined(self):
        """Test MODEL_INFO is defined."""
        if MODEL_INFO is None:
            pytest.skip("MODEL_INFO not implemented yet")
        assert hasattr(MODEL_INFO, 'info')


class TestMetricHelpers:
    """Test metric helper functions."""
    
    def test_get_all_metrics_returns_dict(self):
        """Test get_all_metrics returns a dictionary."""
        metrics = get_all_metrics()
        assert isinstance(metrics, dict)
    
    def test_count_implemented_metrics(self):
        """Test count_implemented_metrics function."""
        implemented, total = count_implemented_metrics()
        assert isinstance(implemented, int)
        assert isinstance(total, int)
        assert implemented >= 0
        assert total > 0
        assert implemented <= total


class TestMetricLabels:
    """Test metric labels configuration."""
    
    def test_request_count_has_correct_labels(self):
        """Test REQUEST_COUNT has method, endpoint, status labels."""
        # Increment with labels to verify they work
        REQUEST_COUNT.labels(method='GET', endpoint='/test', status=200).inc()
    
    def test_request_latency_has_correct_labels(self):
        """Test REQUEST_LATENCY has correct labels."""
        if REQUEST_LATENCY is None:
            pytest.skip("REQUEST_LATENCY not implemented yet")
        REQUEST_LATENCY.labels(method='GET', endpoint='/test').observe(0.1)


class TestMetricsEndpoint:
    """Test the /metrics endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    def test_metrics_endpoint_returns_200(self, client):
        """Test /metrics returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_endpoint_returns_prometheus_format(self, client):
        """Test /metrics returns Prometheus format."""
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"] or \
               "text/plain" in response.headers.get("content-type", "")
    
    def test_metrics_contains_http_requests_total(self, client):
        """Test metrics output contains http_requests_total."""
        response = client.get("/metrics")
        assert "http_requests_total" in response.text


class TestMetricsIntegration:
    """Integration tests for metrics with API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    def test_request_increments_counter(self, client):
        """Test that making requests increments the counter."""
        # Make some requests
        client.get("/health")
        client.get("/health")
        
        # Check metrics
        response = client.get("/metrics")
        assert "http_requests_total" in response.text


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

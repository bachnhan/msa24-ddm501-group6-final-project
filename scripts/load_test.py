"""
Load Testing Script for Customer Churn API.

This script generates load on the API to test metrics collection
and visualization in Grafana.


Usage:
    python scripts/load_test.py
    python scripts/load_test.py --duration 120 --workers 20
"""

import argparse
import random
import time
import concurrent.futures
from typing import Tuple
import requests

# API Configuration
API_URL = "http://localhost:8000"


def make_single_prediction() -> Tuple[bool, float]:
    """
    Make a single prediction request.
    
    Returns:
        Tuple of (success: bool, latency_ms: float)
    """
    # Generate random churn features
    payload = {
        "age": random.randint(18, 70),
        "gender": random.choice(["Male", "Female"]),
        "tenure": random.randint(1, 72),
        "usage_frequency": random.randint(1, 30),
        "support_calls": random.randint(0, 10),
        "payment_delay": random.randint(0, 30),
        "subscription_type": random.choice(["Basic", "Standard", "Premium"]),
        "contract_length": random.choice(["Monthly", "Annual"]),
        "total_spend": round(random.uniform(100.0, 5000.0), 2),
        "last_interaction": random.randint(0, 31)
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=5
        )
        latency_ms = (time.time() - start_time) * 1000
        return response.status_code == 200, latency_ms
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return False, latency_ms


def make_batch_prediction(batch_size: int = 10) -> Tuple[bool, float]:
    """
    Make a batch prediction request.
    
    Args:
        batch_size: Number of predictions in the batch
        
    Returns:
        Tuple of (success: bool, latency_ms: float)
    """
    predictions = []
    for _ in range(batch_size):
        predictions.append({
            "age": random.randint(18, 70),
            "gender": random.choice(["Male", "Female"]),
            "tenure": random.randint(1, 72),
            "usage_frequency": random.randint(1, 30),
            "support_calls": random.randint(0, 10),
            "payment_delay": random.randint(0, 30),
            "subscription_type": random.choice(["Basic", "Standard", "Premium"]),
            "contract_length": random.choice(["Monthly", "Annual"]),
            "total_spend": round(random.uniform(100.0, 5000.0), 2),
            "last_interaction": random.randint(0, 31)
        })
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"predictions": predictions},
            timeout=10
        )
        latency_ms = (time.time() - start_time) * 1000
        return response.status_code == 200, latency_ms
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return False, latency_ms


def check_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False



def run_load_test(duration: int = 60, workers: int = 10, batch_mode: bool = False):
    """
    Run load test for specified duration.
    
    Args:
        duration: Test duration in seconds
        workers: Number of concurrent workers
        batch_mode: If True, use batch predictions
    """
    print("=" * 60)
    print("Load Test for Customer Churn API")
    print("=" * 60)
    print(f"Duration: {duration}s")
    print(f"Workers: {workers}")
    print(f"Mode: {'Batch' if batch_mode else 'Single'}")
    print("=" * 60)
    
    # Check API health
    if not check_health():
        print("ERROR: API is not healthy. Aborting load test.")
        return
    
    # Initialize counters
    total_requests = 0
    successful = 0
    latencies = []
    
    # Run load test using ThreadPoolExecutor
    start_time = time.time()
    last_print_time = start_time
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while time.time() - start_time < duration:
            # Submit tasks
            if batch_mode:
                futures = [executor.submit(make_batch_prediction) for _ in range(workers)]
            else:
                futures = [executor.submit(make_single_prediction) for _ in range(workers)]
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    success, latency = future.result()
                    total_requests += 1
                    if success:
                        successful += 1
                    latencies.append(latency)
                except Exception as e:
                    print(f"Error in request: {e}")
                    total_requests += 1
            
            # Progress update
            elapsed = time.time() - start_time
            if time.time() - last_print_time >= 10:
                print(f"  Progress: {int(elapsed)}s - {total_requests} requests")
                last_print_time = time.time()
            
            # Small delay to control rate
            time.sleep(0.1)
    
    # Print statistics
    print_statistics(total_requests, successful, latencies, duration)


def print_statistics(total: int, successful: int, latencies: list, duration: int):
    """Print load test statistics."""
    import statistics
    
    print("\n" + "=" * 60)
    print("Load Test Results")
    print("=" * 60)
    print(f"Total Requests:    {total}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {total - successful}")
    print(f"Success Rate:      {successful/total*100:.2f}%" if total > 0 else "N/A")
    print(f"Requests/Second:   {total/duration:.2f}")
    
    if latencies:
        print(f"\nLatency Statistics (ms):")
        print(f"  Min:    {min(latencies):.2f}")
        print(f"  Max:    {max(latencies):.2f}")
        print(f"  Mean:   {statistics.mean(latencies):.2f}")
        print(f"  Median: {statistics.median(latencies):.2f}")
        if len(latencies) > 1:
            print(f"  StdDev: {statistics.stdev(latencies):.2f}")
        
        # Percentiles
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.5)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        print(f"  P50:    {p50:.2f}")
        print(f"  P95:    {p95:.2f}")
        print(f"  P99:    {p99:.2f}")
    
    print("=" * 60)



def run_variable_load(duration: int = 120, max_workers: int = 50):
    """
    Run load test with variable load pattern.
    
    This tests how the system handles varying load.
    """
    print("=" * 60)
    print("Variable Load Test")
    print("=" * 60)
    
    start_time = time.time()
    ramp_duration = 30
    peak_duration = 60
    
    total_requests = 0
    successful = 0
    latencies = []
    
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        
        # Calculate current workers based on ramp
        if elapsed < ramp_duration:
            # Ramp up
            current_workers = int(1 + (max_workers - 1) * (elapsed / ramp_duration))
        elif elapsed < ramp_duration + peak_duration:
            # Maintain peak
            current_workers = max_workers
        else:
            # Ramp down
            rem_time = duration - elapsed
            current_workers = int(1 + (max_workers - 1) * (rem_time / ramp_duration))
            current_workers = max(1, current_workers)
        
        print(f"  [{int(elapsed)}s] Workers: {current_workers}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=current_workers) as executor:
            futures = [executor.submit(make_single_prediction) for _ in range(current_workers)]
            for future in concurrent.futures.as_completed(futures):
                success, latency = future.result()
                total_requests += 1
                if success:
                    successful += 1
                latencies.append(latency)
        
        time.sleep(1)
    
    print_statistics(total_requests, successful, latencies, duration)



def run_spike_test(normal_workers: int = 5, spike_workers: int = 100, spike_duration: int = 10):
    """
    Run spike test to test system resilience.
    """
    print("=" * 60)
    print("Spike Test")
    print("=" * 60)
    
    duration = 70 # 30 + 10 + 30
    start_time = time.time()
    
    total_requests = 0
    successful = 0
    latencies = []
    
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        
        if 30 <= elapsed < 30 + spike_duration:
            current_workers = spike_workers
            print(f"  [{int(elapsed)}s] SPIKE! Workers: {current_workers}")
        else:
            current_workers = normal_workers
            print(f"  [{int(elapsed)}s] Normal. Workers: {current_workers}")
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=current_workers) as executor:
            futures = [executor.submit(make_single_prediction) for _ in range(current_workers)]
            for future in concurrent.futures.as_completed(futures):
                success, latency = future.result()
                total_requests += 1
                if success:
                    successful += 1
                latencies.append(latency)
        
        time.sleep(1)
        
    print_statistics(total_requests, successful, latencies, duration)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Load test the Customer Churn API")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--batch", action="store_true", help="Use batch predictions")
    parser.add_argument("--variable", action="store_true", help="Run variable load pattern")
    parser.add_argument("--spike", action="store_true", help="Run spike test")
    
    args = parser.parse_args()
    
    if args.variable:
        run_variable_load(args.duration, args.workers)
    elif args.spike:
        run_spike_test(normal_workers=args.workers)
    else:
        run_load_test(args.duration, args.workers, args.batch)


if __name__ == "__main__":
    main()

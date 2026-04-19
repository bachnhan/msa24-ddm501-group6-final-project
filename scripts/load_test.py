import requests
import time
import random
import concurrent.futures
from statistics import mean, quantiles

# --- CONFIGURATION ---
URL = "https://customer-churn-api-3p4j.onrender.com/predict"
TOTAL_REQUESTS = 200
CONCURRENT_USERS = 10
# ---------------------


def generate_random_customer():
    """Generates random Telco-style data for prediction."""
    monthly_charges = round(random.uniform(20, 120), 2)
    tenure = random.randint(1, 72)
    return {
        "gender": random.choice(["Male", "Female"]),
        "seniorcitizen": random.choice([0, 1]),
        "partner": random.choice(["Yes", "No"]),
        "dependents": random.choice(["Yes", "No"]),
        "tenure": tenure,
        "phoneservice": random.choice(["Yes", "No"]),
        "multiplelines": random.choice(["No phone service", "No", "Yes"]),
        "internetservice": random.choice(["DSL", "Fiber optic", "No"]),
        "onlinesecurity": random.choice(["No", "Yes", "No internet service"]),
        "onlinebackup": random.choice(["No", "Yes", "No internet service"]),
        "deviceprotection": random.choice(["No", "Yes", "No internet service"]),
        "techsupport": random.choice(["No", "Yes", "No internet service"]),
        "streamingtv": random.choice(["No", "Yes", "No internet service"]),
        "streamingmovies": random.choice(["No", "Yes", "No internet service"]),
        "contract": random.choice(["Month-to-month", "One year", "Two year"]),
        "paperlessbilling": random.choice(["Yes", "No"]),
        "paymentmethod": random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
        ),
        "monthlycharges": monthly_charges,
        "totalcharges": round(monthly_charges * tenure, 2),
    }


def send_request():
    """Sends a single POST request and measures latency."""
    payload = generate_random_customer()
    start = time.perf_counter()
    try:
        response = requests.post(URL, json=payload, timeout=10)
        latency = (time.perf_counter() - start) * 1000
        return response.status_code, latency
    except Exception as e:
        return 500, 0


def run_load_test():
    print(
        f"🚀 Starting Load Test: {TOTAL_REQUESTS} requests, {CONCURRENT_USERS} concurrent users..."
    )
    print(f"🔗 Target URL: {URL}\n")

    results = []
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=CONCURRENT_USERS
    ) as executor:
        futures = [executor.submit(send_request) for _ in range(TOTAL_REQUESTS)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    total_time = time.perf_counter() - start_time

    # Process Results
    latencies = [r[1] for r in results if r[0] == 200]
    success_count = sum(1 for r in results if r[0] == 200)
    error_count = TOTAL_REQUESTS - success_count

    if not latencies:
        print("❌ All requests failed. Check if API is running.")
        return

    print("=" * 40)
    print("🚦 LOAD TEST RESULTS")
    print("=" * 40)
    print(f"✅ Successful Requests: {success_count}")
    print(f"❌ Failed Requests    : {error_count}")
    print(f"⏱️  Total Duration     : {total_time:.2f} seconds")
    print(f"📈 Throughput         : {success_count / total_time:.2f} req/s")
    print("-" * 40)
    print(f"平均 (Average) Latency: {mean(latencies):.2f} ms")

    if len(latencies) >= 2:
        q = quantiles(latencies, n=100)
        print(f"⚡ P50 (Median)   : {q[49]:.2f} ms")
        print(f"⚡ P95 (Tail)     : {q[94]:.2f} ms")
        print(f"⚡ P99 (Worst)    : {q[98]:.2f} ms")
    print("=" * 40)


if __name__ == "__main__":
    run_load_test()

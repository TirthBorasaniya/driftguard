"""Locust load test: measures p50/p95/p99 latency for /predict endpoint."""

from locust import HttpUser, between, task


SAMPLE_PAYLOAD = {
    "cc_num": "4532015112830366",
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "misc_net",
    "amt": 149.62,
    "gender": "F",
    "city": "Henderson",
    "state": "TX",
    "zip": "76054",
    "lat": 36.0788,
    "long": -81.1781,
    "city_pop": 35550,
    "job": "Scientist, product/process development",
    "dob": "1987-01-01",
    "merch_lat": 36.011293,
    "merch_long": -82.048315,
    "trans_date_trans_time": "2020-06-21 12:14:25",
}


class FraudUser(HttpUser):
    wait_time = between(0.05, 0.1)

    @task(10)
    def predict(self):
        self.client.post("/predict", json=SAMPLE_PAYLOAD)

    @task(1)
    def health(self):
        self.client.get("/health")

"""Locust load test: measures p50/p95/p99 latency for the /predict endpoint."""

from locust import HttpUser, between, task


SAMPLE_PAYLOAD = {
    "flow_duration": 100000.0,
    "flow_bytes_per_sec": 5000.0,
    "flow_packets_per_sec": 50.0,
    "total_fwd_packets": 10.0,
    "total_bwd_packets": 8.0,
    "packet_length_mean": 120.0,
    "packet_length_std": 30.0,
    "flow_iat_mean": 2000.0,
    "syn_flag_count": 1.0,
    "src_ip": "192.168.10.5",
    "flow_id": "192.168.10.5-52.6.13.28-49158-443-6",
}


class NetworkFlowUser(HttpUser):
    wait_time = between(0.05, 0.1)

    @task(10)
    def predict(self):
        self.client.post("/predict", json=SAMPLE_PAYLOAD)

    @task(1)
    def health(self):
        self.client.get("/health")

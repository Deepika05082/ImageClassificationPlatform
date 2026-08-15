import time
import logging

logging.basicConfig(level=logging.INFO)

class Monitor:
    def __init__(self):
        self.request_count = 0
        self.total_latency = 0.0

    def log_request(self, latency: float):
        self.request_count += 1
        self.total_latency += latency
        avg_latency = self.total_latency / self.request_count
        logging.info(f"Requests={self.request_count}, Last latency={latency:.3f}s, Avg latency={avg_latency:.3f}s")

"""
test_edge_cases.py - Edge Case and Worker Resilience Verification

Validates the complete set of required edge cases against live TorchServe:
1. Empty prompt
2. Missing prompt
3. Non-string prompt
4. Whitespace-only prompt
5. Prompt over 500 characters
6. Out-of-range temperature
7. Out-of-range max_new_tokens
8. Worker survival: Valid request right after an invalid one
9. Latency benchmark: 5 consecutive requests (warm-up reported separately)
"""

from __future__ import annotations

import os
import time
import unittest
from typing import Any, Dict, List

import requests

from generation_config import MAX_PROMPT_CHARS

DEFAULT_TS_URL = os.environ.get("TS_URL", "http://127.0.0.1:8085/predictions/story_model")


def is_server_online(endpoint_url: str) -> bool:
    """Checks whether the TorchServe ping endpoint is reachable."""
    try:
        parts = endpoint_url.split("/predictions")[0]
        ping_url = f"{parts}/ping"
        resp = requests.get(ping_url, timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


class TestEdgeCasesAndResilience(unittest.TestCase):
    """Verifies edge cases, worker error survival, and sequential latency."""

    endpoint_url: str = DEFAULT_TS_URL

    @classmethod
    def setUpClass(cls) -> None:
        if not is_server_online(cls.endpoint_url):
            fallback_url = "http://127.0.0.1:8080/predictions/story_model"
            if is_server_online(fallback_url):
                cls.endpoint_url = fallback_url
            else:
                raise unittest.SkipTest(
                    f"TorchServe server is not reachable at '{cls.endpoint_url}'. "
                    "Start TorchServe before running edge case tests."
                )

    def _post(self, payload: Any) -> requests.Response:
        return requests.post(
            self.endpoint_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )

    # 1. Empty prompt
    def test_edge_case_empty_prompt(self) -> None:
        resp = self._post({"prompt": ""})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("error"), "prompt is required")

    # 2. Missing prompt
    def test_edge_case_missing_prompt(self) -> None:
        resp = self._post({"max_new_tokens": 40})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("error"), "prompt is required")

    # 3. Non-string prompt
    def test_edge_case_non_string_prompt(self) -> None:
        resp = self._post({"prompt": 12345})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("error"), "prompt must be a string")

    # 4. Whitespace-only prompt
    def test_edge_case_whitespace_prompt(self) -> None:
        resp = self._post({"prompt": "   \n\t  \r  "})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("error"), "prompt is required")

    # 5. Prompt over 500 characters
    def test_edge_case_prompt_over_500_chars(self) -> None:
        resp = self._post({"prompt": "Word " * 110})  # 550 characters
        self.assertEqual(resp.status_code, 200)
        self.assertIn("prompt too long", resp.json().get("error", ""))

    # 6. Out-of-range temperature
    def test_edge_case_out_of_range_temperature(self) -> None:
        resp_high = self._post({"prompt": "Valid prompt", "temperature": 2.5})
        self.assertEqual(resp_high.status_code, 200)
        self.assertIn("temperature must be between", resp_high.json().get("error", ""))

        resp_low = self._post({"prompt": "Valid prompt", "temperature": 0.05})
        self.assertEqual(resp_low.status_code, 200)
        self.assertIn("temperature must be between", resp_low.json().get("error", ""))

    # 7. Out-of-range max_new_tokens
    def test_edge_case_out_of_range_max_new_tokens(self) -> None:
        resp_high = self._post({"prompt": "Valid prompt", "max_new_tokens": 500})
        self.assertEqual(resp_high.status_code, 200)
        self.assertIn("max_new_tokens must be between", resp_high.json().get("error", ""))

        resp_low = self._post({"prompt": "Valid prompt", "max_new_tokens": 5})
        self.assertEqual(resp_low.status_code, 200)
        self.assertIn("max_new_tokens must be between", resp_low.json().get("error", ""))

    # 8. Worker survival: Valid request right after an invalid one
    def test_worker_survival_after_invalid_request(self) -> None:
        # Step A: Send an invalid request that triggers validation error
        invalid_resp = self._post({"prompt": "", "max_new_tokens": 9999})
        self.assertEqual(invalid_resp.status_code, 200)
        self.assertIn("error", invalid_resp.json())

        # Step B: Immediately send a valid request
        valid_resp = self._post({
            "prompt": "The observatory dish rotated smoothly toward the horizon...",
            "max_new_tokens": 30,
            "seed": 42,
        })
        self.assertEqual(valid_resp.status_code, 200)
        data = valid_resp.json()
        self.assertNotIn("error", data)
        self.assertIn("generated_text", data)
        self.assertTrue(len(data["generated_text"]) > 0)
        self.assertEqual(data["model"], "distilgpt2")

    # 9. Latency benchmark: 5 consecutive requests (warm-up reported separately)
    def test_five_consecutive_requests_latency(self) -> None:
        test_prompt = "Beneath the glowing crystal spires of the city,"

        # 1. Warm-up request
        t0 = time.time()
        warmup_resp = self._post({"prompt": test_prompt, "max_new_tokens": 40})
        warmup_latency = round((time.time() - t0) * 1000.0, 2)
        self.assertEqual(warmup_resp.status_code, 200)
        self.assertIn("generated_text", warmup_resp.json())

        # 2. Five consecutive requests
        latencies: List[float] = []
        for i in range(5):
            t_start = time.time()
            resp = self._post({"prompt": test_prompt, "max_new_tokens": 40})
            lat = round((time.time() - t_start) * 1000.0, 2)
            latencies.append(lat)
            self.assertEqual(resp.status_code, 200)
            self.assertIn("generated_text", resp.json())

        min_lat = min(latencies)
        avg_lat = round(sum(latencies) / len(latencies), 2)
        max_lat = max(latencies)

        print("\n" + "=" * 60)
        print("LATENCY BENCHMARK (5 Consecutive Requests):")
        print(f"Warm-up Request Latency: {warmup_latency:.2f} ms")
        print(f"Subsequent Latencies:    {latencies}")
        print(f"Min Latency:             {min_lat:.2f} ms")
        print(f"Avg Latency:             {avg_lat:.2f} ms")
        print(f"Max Latency:             {max_lat:.2f} ms")
        print("=" * 60)

        # Store on class for reference in reports
        TestEdgeCasesAndResilience.warmup_latency = warmup_latency
        TestEdgeCasesAndResilience.min_latency = min_lat
        TestEdgeCasesAndResilience.avg_latency = avg_lat
        TestEdgeCasesAndResilience.max_latency = max_lat


if __name__ == "__main__":
    unittest.main()

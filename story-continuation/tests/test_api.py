"""
test_api.py - Integration Test Suite for TorchServe Story Continuation Endpoint

Sends HTTP requests against a live TorchServe deployment, benchmarks latency,
validates the API contract across assignment cases, length variations, and error inputs,
and exports granular records into `results/test_results.csv`.
"""

from __future__ import annotations

import csv
import datetime
import os
import time
import unittest
from typing import Any, Dict, List, Optional

import requests

from generation_config import MAX_PROMPT_CHARS

# Default endpoint URL (allows override via TS_URL environment variable)
DEFAULT_TS_URL = os.environ.get("TS_URL", "http://127.0.0.1:8085/predictions/story_model")
RESULTS_CSV_PATH = os.path.join("results", "test_results.csv")


def is_server_online(endpoint_url: str) -> bool:
    """Checks whether the TorchServe ping endpoint is reachable."""
    try:
        # Extract host:port and append /ping
        parts = endpoint_url.split("/predictions")[0]
        ping_url = f"{parts}/ping"
        resp = requests.get(ping_url, timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


class TestStoryContinuationAPI(unittest.TestCase):
    """Integration test suite executing requests against live TorchServe."""

    endpoint_url: str = DEFAULT_TS_URL
    records: List[Dict[str, Any]] = []
    first_request_latency: Optional[float] = None
    subsequent_latencies: List[float] = []

    @classmethod
    def setUpClass(cls) -> None:
        """Verify server connectivity before running test suite."""
        # Check primary endpoint or fallback to 8080 if primary unreachable
        if not is_server_online(cls.endpoint_url):
            fallback_url = "http://127.0.0.1:8080/predictions/story_model"
            if is_server_online(fallback_url):
                cls.endpoint_url = fallback_url
            else:
                raise unittest.SkipTest(
                    f"TorchServe server is not reachable at '{cls.endpoint_url}'. "
                    "Start TorchServe before running integration tests."
                )

        os.makedirs("results", exist_ok=True)
        cls.records = []
        cls.first_request_latency = None
        cls.subsequent_latencies = []

    def _send_request(self, payload: Any) -> requests.Response:
        """Sends POST request to endpoint, logs latency and metrics."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        start = time.time()
        try:
            resp = requests.post(
                self.endpoint_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            latency_ms = round((time.time() - start) * 1000.0, 2)
            status_code = resp.status_code
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text}
        except Exception as exc:
            latency_ms = round((time.time() - start) * 1000.0, 2)
            status_code = 0
            body = {"error": f"request exception: {str(exc)}"}

        if self.__class__.first_request_latency is None:
            self.__class__.first_request_latency = latency_ms
        else:
            self.__class__.subsequent_latencies.append(latency_ms)

        prompt_str = ""
        settings_str = ""
        generated_text = ""
        error_msg = ""

        if isinstance(payload, dict):
            prompt_str = str(payload.get("prompt", ""))
            settings_str = str({k: v for k, v in payload.items() if k != "prompt"})
        else:
            prompt_str = str(payload)

        if isinstance(body, dict):
            generated_text = body.get("generated_text", "")
            error_msg = body.get("error", "")

        record_entry = {
            "timestamp": timestamp,
            "prompt_preview": prompt_str[:50].replace("\n", " "),
            "settings": settings_str,
            "generated_text": generated_text[:120].replace("\n", " "),
            "word_count": len(generated_text.split()) if generated_text else 0,
            "latency_ms": latency_ms,
            "http_status": status_code,
            "error": error_msg,
        }
        self.__class__.records.append(record_entry)
        return resp

    # =========================================================================
    # 1. Assignment Core Cases
    # =========================================================================
    def test_assignment_case_1_short(self) -> None:
        """Case 1: Mystery box with Short length (40 tokens)."""
        prompt = "One day, a student discovered a mysterious box in the college library..."
        resp = self._send_request({"prompt": prompt, "max_new_tokens": 40})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("generated_text", data)
        self.assertTrue(len(data["generated_text"]) > 0)
        self.assertEqual(data["model"], "distilgpt2")

    def test_assignment_case_2_long(self) -> None:
        """Case 2: Robot classroom with Long length (150 tokens)."""
        prompt = "A robot entered the classroom for the first time..."
        resp = self._send_request({"prompt": prompt, "max_new_tokens": 150})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("generated_text", data)
        self.assertTrue(len(data["generated_text"]) > 0)

    def test_assignment_case_3_temperature_comparison(self) -> None:
        """Case 3: Dark rainy night at temp 0.3 and 1.2 with identical seed."""
        prompt = "It was a dark and rainy night..."
        r_low = self._send_request({"prompt": prompt, "max_new_tokens": 60, "temperature": 0.3, "seed": 99})
        r_high = self._send_request({"prompt": prompt, "max_new_tokens": 60, "temperature": 1.2, "seed": 99})

        self.assertEqual(r_low.status_code, 200)
        self.assertEqual(r_high.status_code, 200)
        self.assertIn("generated_text", r_low.json())
        self.assertIn("generated_text", r_high.json())

    # =========================================================================
    # 2. 10 Prompts x 2 Lengths (40 and 120 tokens) = 20 Test Runs
    # =========================================================================
    def test_ten_prompts_at_two_lengths(self) -> None:
        prompts = [
            "In an overgrown botanical garden, a strange mechanical flower began to",
            "The captain looked through the telescope and realized the island was",
            "Deep inside the abandoned subway tunnel, graffiti covered a door that",
            "Every year on the winter solstice, the clock in the village square would",
            "The astronaut reached for the satellite antenna, unaware that something",
            "Behind the bookshelf in the grandfather's study was an old brass key and",
            "During the heavy thunder storm, the computer screen suddenly displayed",
            "The alchemist carefully poured the emerald potion into the flask, expecting",
            "At the edge of the desert, travellers found a city carved entirely from",
            "The newly activated synthetic android opened its optical sensors and whispered",
        ]

        for p in prompts:
            for tokens in (40, 120):
                resp = self._send_request({"prompt": p, "max_new_tokens": tokens, "temperature": 0.8})
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertIn("generated_text", data)
                self.assertTrue(len(data["generated_text"]) > 0)

    # =========================================================================
    # 3. Comprehensive Error Handling Cases
    # =========================================================================
    def test_error_empty_prompt(self) -> None:
        resp = self._send_request({"prompt": ""})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("error"), "prompt is required")

    def test_error_missing_prompt(self) -> None:
        resp = self._send_request({"max_new_tokens": 50})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("error"), "prompt is required")

    def test_error_non_string_prompt(self) -> None:
        resp = self._send_request({"prompt": 98765})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("error"), "prompt must be a string")

    def test_error_whitespace_prompt(self) -> None:
        resp = self._send_request({"prompt": "    \t\n   "})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("error"), "prompt is required")

    def test_error_prompt_over_max_chars(self) -> None:
        resp = self._send_request({"prompt": "A" * (MAX_PROMPT_CHARS + 5)})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("prompt too long", resp.json().get("error", ""))

    def test_error_out_of_range_temperature(self) -> None:
        resp = self._send_request({"prompt": "Valid prompt", "temperature": 2.5})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("temperature must be between", resp.json().get("error", ""))

    def test_error_out_of_range_max_new_tokens(self) -> None:
        resp = self._send_request({"prompt": "Valid prompt", "max_new_tokens": 500})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("max_new_tokens must be between", resp.json().get("error", ""))

    @classmethod
    def tearDownClass(cls) -> None:
        """Export all request records to CSV and print summary performance table."""
        if not cls.records:
            return

        # Write to results/test_results.csv
        fieldnames = [
            "timestamp",
            "prompt_preview",
            "settings",
            "generated_text",
            "word_count",
            "latency_ms",
            "http_status",
            "error",
        ]
        with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cls.records)

        # Print performance summary
        total_runs = len(cls.records)
        error_runs = sum(1 for r in cls.records if r["error"])
        success_runs = total_runs - error_runs

        latencies = [r["latency_ms"] for r in cls.records]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        min_lat = min(latencies) if latencies else 0
        max_lat = max(latencies) if latencies else 0

        sub_avg = (
            sum(cls.subsequent_latencies) / len(cls.subsequent_latencies)
            if cls.subsequent_latencies
            else 0
        )

        print("\n" + "=" * 70)
        print("TORCHSERVE INTEGRATION TEST BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"Total Requests Executed:    {total_runs}")
        print(f"Successful Generations:     {success_runs}")
        print(f"Expected Error Responses:   {error_runs}")
        print("-" * 70)
        print(f"Warm-up Request Latency:    {cls.first_request_latency:.2f} ms")
        print(f"Subsequent Average Latency: {sub_avg:.2f} ms")
        print(f"Min Latency / Max Latency:  {min_lat:.2f} ms / {max_lat:.2f} ms")
        print(f"Detailed CSV exported to:   {RESULTS_CSV_PATH} ({total_runs} rows)")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    unittest.main()

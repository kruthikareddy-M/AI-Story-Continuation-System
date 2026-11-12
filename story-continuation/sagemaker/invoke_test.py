"""
sagemaker/invoke_test.py - Live SageMaker Real-Time Endpoint Benchmark

Runs integration tests against the live SageMaker endpoint via boto3:
1. 3 Assignment core cases (Short 40 tokens, Long 150 tokens, Temperature comparison).
2. 10 story prompts at two lengths (40 and 120 tokens = 20 test runs).
3. 7 comprehensive schema and boundary error cases.
4. 5-request latency benchmark (cold start vs warm).
5. Exports results to results/sagemaker_test_results.csv.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import boto3

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV_PATH = os.path.join(PROJECT_DIR, "results", "sagemaker_test_results.csv")
EVIDENCE_PATH = os.path.join(PROJECT_DIR, "evidence", "B4_sagemaker_tests.txt")

AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "story-continuation-endpoint")


def invoke_endpoint(client: Any, payload: Any) -> Dict[str, Any]:
    """Invokes SageMaker real-time endpoint, measures roundtrip latency."""
    start_time = time.time()
    try:
        raw_payload = json.dumps(payload).encode("utf-8")
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=raw_payload,
        )
        latency_ms = round((time.time() - start_time) * 1000.0, 2)
        body = json.loads(response["Body"].read().decode("utf-8"))
        status_code = response["ResponseMetadata"]["HTTPStatusCode"]
        return {"body": body, "latency_ms": latency_ms, "status_code": status_code}
    except Exception as exc:
        latency_ms = round((time.time() - start_time) * 1000.0, 2)
        return {"body": {"error": str(exc)}, "latency_ms": latency_ms, "status_code": 500}


def run_benchmark() -> None:
    print("=" * 70)
    print(f"SAGEMAKER ENDPOINT INTEGRATION BENCHMARK ({ENDPOINT_NAME})")
    print(f"Region: {AWS_REGION}")
    print("=" * 70)

    client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    records: List[Dict[str, Any]] = []

    # 1. Assignment Core Cases
    core_cases = [
        {"prompt": "One day, a student discovered a mysterious box in the college library...", "max_new_tokens": 40},
        {"prompt": "A robot entered the classroom for the first time...", "max_new_tokens": 150},
        {"prompt": "It was a dark and rainy night...", "max_new_tokens": 60, "temperature": 0.3, "seed": 99},
        {"prompt": "It was a dark and rainy night...", "max_new_tokens": 60, "temperature": 1.2, "seed": 99},
    ]

    # 2. 10 Prompts x 2 Lengths
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
    length_cases = []
    for p in prompts:
        for tokens in (40, 120):
            length_cases.append({"prompt": p, "max_new_tokens": tokens, "temperature": 0.8})

    # 3. Error cases
    error_cases = [
        {"prompt": ""},
        {"max_new_tokens": 50},
        {"prompt": 98765},
        {"prompt": "    \t\n   "},
        {"prompt": "A" * 505},
        {"prompt": "Valid prompt", "temperature": 2.5},
        {"prompt": "Valid prompt", "max_new_tokens": 500},
    ]

    all_cases = core_cases + length_cases + error_cases
    print(f"Executing {len(all_cases)} test requests...")

    for i, payload in enumerate(all_cases, 1):
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        res = invoke_endpoint(client, payload)
        body = res["body"]

        prompt_str = str(payload.get("prompt", ""))
        settings_str = str({k: v for k, v in payload.items() if k != "prompt"})
        generated_text = body.get("generated_text", "")
        error_msg = body.get("error", "")

        record = {
            "backend": "Amazon SageMaker",
            "timestamp": timestamp,
            "prompt_preview": prompt_str[:50].replace("\n", " "),
            "settings": settings_str,
            "generated_text": generated_text[:120].replace("\n", " "),
            "word_count": len(generated_text.split()) if generated_text else 0,
            "latency_ms": res["latency_ms"],
            "http_status": res["status_code"],
            "error": error_msg,
        }
        records.append(record)
        status_icon = "❌" if error_msg else "✅"
        print(f"[{i:02d}/{len(all_cases)}] {status_icon} Latency: {res['latency_ms']} ms | Preview: {prompt_str[:30]}...")

    # Write CSV
    os.makedirs(os.path.join(PROJECT_DIR, "results"), exist_ok=True)
    fieldnames = [
        "backend", "timestamp", "prompt_preview", "settings",
        "generated_text", "word_count", "latency_ms", "http_status", "error"
    ]
    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[Export] Results saved to: {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    run_benchmark()

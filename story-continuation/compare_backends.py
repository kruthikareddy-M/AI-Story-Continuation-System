"""
compare_backends.py - Multi-Backend Inference Comparator

Evaluates and compares execution across:
1. Local PyTorch (local_inference.py)
2. Local TorchServe (REST API)
3. Amazon SageMaker (skipped if SAGEMAKER_ENDPOINT_NAME not set)

Saves comparison metrics to `results/comparison.csv`.
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from local_inference import generate_story

TS_URL = os.environ.get("TS_URL", "http://127.0.0.1:8085/predictions/story_model")
SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
COMPARISON_CSV_PATH = os.path.join("results", "comparison.csv")

BENCHMARK_PROMPT = "One day, a student discovered a mysterious box in the college library..."
BENCHMARK_SETTINGS = {
    "max_new_tokens": 60,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "seed": 42,
}
EXPECTED_KEYS = {"prompt", "generated_text", "settings", "model"}


def run_local_pytorch() -> Dict[str, Any]:
    """Runs inference via local PyTorch."""
    start = time.time()
    res = generate_story(prompt=BENCHMARK_PROMPT, **BENCHMARK_SETTINGS)
    latency_ms = round((time.time() - start) * 1000.0, 2)
    res["latency_ms"] = latency_ms
    return res


def run_torchserve(url: str = TS_URL) -> Dict[str, Any]:
    """Runs inference via TorchServe REST endpoint."""
    payload = {"prompt": BENCHMARK_PROMPT, **BENCHMARK_SETTINGS}
    start = time.time()
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
        latency_ms = round((time.time() - start) * 1000.0, 2)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            data["latency_ms"] = latency_ms
            return data
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}", "latency_ms": latency_ms}
    except Exception as exc:
        latency_ms = round((time.time() - start) * 1000.0, 2)
        return {"error": f"Connection failed: {exc}", "latency_ms": latency_ms}


def run_sagemaker() -> Dict[str, Any]:
    """Runs inference via Amazon SageMaker runtime endpoint if configured."""
    if not SAGEMAKER_ENDPOINT_NAME:
        return {"error": "Skipped: SAGEMAKER_ENDPOINT_NAME env var not set (Part B)", "latency_ms": 0.0}

    payload = {"prompt": BENCHMARK_PROMPT, **BENCHMARK_SETTINGS}
    start = time.time()
    try:
        import boto3
        client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
        resp = client.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload),
        )
        latency_ms = round((time.time() - start) * 1000.0, 2)
        data = json.loads(resp["Body"].read().decode("utf-8"))
        data["latency_ms"] = latency_ms
        return data
    except Exception as exc:
        latency_ms = round((time.time() - start) * 1000.0, 2)
        return {"error": f"SageMaker invocation error: {exc}", "latency_ms": latency_ms}


def compare() -> None:
    os.makedirs("results", exist_ok=True)

    print("\n" + "=" * 75)
    print("AI STORY CONTINUATION - BACKEND COMPARISON BENCHMARK")
    print("=" * 75)
    print(f"Benchmark Prompt: '{BENCHMARK_PROMPT}'")
    print(f"Settings:         {BENCHMARK_SETTINGS}")
    print("-" * 75)
    print("NOTE: With sampling, exact wording may differ slightly across different execution")
    print("environments. This benchmark compares execution status, latency, contract compliance,")
    print("and settings reproduction across backends.")
    print("-" * 75)

    results: List[Dict[str, Any]] = []

    # 1. Local PyTorch
    print("Running Backend 1: Local PyTorch...")
    pt_res = run_local_pytorch()

    # 2. Local TorchServe
    print(f"Running Backend 2: Local TorchServe ({TS_URL})...")
    ts_res = run_torchserve()

    # 3. Amazon SageMaker
    print("Running Backend 3: Amazon SageMaker...")
    sm_res = run_sagemaker()

    # Evaluate outputs
    backends = [
        ("Local PyTorch", pt_res),
        ("Local TorchServe", ts_res),
        ("Amazon SageMaker", sm_res),
    ]

    csv_rows = []
    print("\n" + f"{'Backend':<20} | {'Status':<10} | {'Keys Match':<11} | {'Latency':<10} | {'Text Preview (80 chars)'}")
    print("-" * 85)

    for name, res in backends:
        has_error = "error" in res
        status = "FAILED" if (has_error and "Skipped" not in res["error"]) else ("SKIPPED" if has_error and "Skipped" in res["error"] else "SUCCESS")
        keys_match = (EXPECTED_KEYS.issubset(set(res.keys()))) if status == "SUCCESS" else False
        latency_str = f"{res.get('latency_ms', 0):.1f} ms"
        text_preview = res.get("generated_text", res.get("error", ""))[:80].replace("\n", " ")

        print(f"{name:<20} | {status:<10} | {str(keys_match):<11} | {latency_str:<10} | {text_preview}")

        csv_rows.append({
            "backend": name,
            "status": status,
            "keys_match": keys_match,
            "settings_echoed": str(res.get("settings", "")),
            "latency_ms": res.get("latency_ms", 0),
            "text_preview": text_preview,
        })

    # Save to CSV
    with open(COMPARISON_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "status", "keys_match", "settings_echoed", "latency_ms", "text_preview"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print("-" * 85)
    print(f"Comparison report saved to: {COMPARISON_CSV_PATH}")

    # Check identical seeded text between PyTorch and TorchServe
    pt_text = pt_res.get("generated_text", "")
    ts_text = ts_res.get("generated_text", "")
    print("\nSeeded Output Alignment Analysis (PyTorch CPU vs TorchServe CPU, seed=42):")
    if pt_text and ts_text:
        if pt_text == ts_text:
            print("MATCH: Local PyTorch and TorchServe outputs are 100% IDENTICAL.")
        else:
            print("VARIATION DETECTED: Outputs differ slightly in wording or token sequence.")
    else:
        print("Could not compare text outputs (one or more backends did not return generated text).")
    print("=" * 75 + "\n")


def main() -> None:
    compare()


if __name__ == "__main__":
    main()

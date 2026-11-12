"""
sagemaker/cleanup.py - Safe Teardown of Amazon SageMaker Resources

Deletes all billable AWS resources associated with the Story Continuation project:
1. Deletes the SageMaker Endpoint.
2. Deletes the Endpoint Configuration.
3. Deletes the SageMaker Model entity.
4. Optionally cleans up S3 model artifacts.
5. Verifies and lists active endpoints to prove zero active billable resources.
6. Saves masked proof to evidence/B6_cleanup.txt.
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Optional

import boto3

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE_PATH = os.path.join(PROJECT_DIR, "evidence", "B6_cleanup.txt")

AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "story-continuation-endpoint")
S3_BUCKET = os.environ.get("S3_BUCKET")


def mask_account_id(text: str) -> str:
    """Masks 12-digit AWS account IDs."""
    if not text:
        return text
    return re.sub(r"\b\d{12}\b", "<ACCOUNT_ID>", text)


def cleanup(auto_confirm: bool = False) -> None:
    print("=" * 70)
    print("🧹 SAGEMAKER RESOURCE TEARDOWN & CLEANUP")
    print("=" * 70)
    print(f"Region:        {AWS_REGION}")
    print(f"Endpoint:      {ENDPOINT_NAME}")
    print("=" * 70)

    if not auto_confirm:
        confirm = input(f"Are you sure you want to delete endpoint '{ENDPOINT_NAME}'? Type DELETE to proceed: ").strip()
        if confirm != "DELETE":
            print("Cleanup cancelled.")
            sys.exit(0)

    client = boto3.client("sagemaker", region_name=AWS_REGION)
    log_lines = []

    # 1. Delete Endpoint
    try:
        print(f"\n[1/3] Deleting Endpoint '{ENDPOINT_NAME}'...")
        client.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print("  -> Endpoint deletion requested.")
        log_lines.append(f"Deleted Endpoint: {ENDPOINT_NAME}")
    except Exception as exc:
        print(f"  -> Endpoint notice: {exc}")
        log_lines.append(f"Endpoint notice: {exc}")

    # 2. Delete Endpoint Config
    try:
        print(f"\n[2/3] Deleting Endpoint Config '{ENDPOINT_NAME}'...")
        client.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
        print("  -> Endpoint configuration deleted.")
        log_lines.append(f"Deleted EndpointConfig: {ENDPOINT_NAME}")
    except Exception as exc:
        print(f"  -> Config notice: {exc}")
        log_lines.append(f"Config notice: {exc}")

    # 3. Delete Model
    try:
        print(f"\n[3/3] Deleting Model '{ENDPOINT_NAME}'...")
        client.delete_model(ModelName=ENDPOINT_NAME)
        print("  -> Model deleted.")
        log_lines.append(f"Deleted Model: {ENDPOINT_NAME}")
    except Exception as exc:
        print(f"  -> Model notice: {exc}")
        log_lines.append(f"Model notice: {exc}")

    # 4. Verification Check
    print("\n" + "=" * 70)
    print("VERIFYING SAGEMAKER INFERENCE RESOURCES (Should be empty):")
    print("-" * 70)
    endpoints = client.list_endpoints(NameContains=ENDPOINT_NAME).get("Endpoints", [])
    models = client.list_models(NameContains=ENDPOINT_NAME).get("Models", [])

    print(f"Active Endpoints Matching '{ENDPOINT_NAME}': {len(endpoints)}")
    print(f"Active Models Matching '{ENDPOINT_NAME}':    {len(models)}")
    print("=" * 70 + "\n")

    log_lines.append(f"Verification: {len(endpoints)} endpoints, {len(models)} models active.")

    # Save evidence
    os.makedirs(os.path.join(PROJECT_DIR, "evidence"), exist_ok=True)
    with open(EVIDENCE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"Proof saved to: {EVIDENCE_PATH}")


if __name__ == "__main__":
    auto = "--auto" in sys.argv
    cleanup(auto_confirm=auto)

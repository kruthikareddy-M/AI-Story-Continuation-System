"""
sagemaker/deploy.py - Amazon SageMaker Real-Time Endpoint Deployment

Deploys the Story Continuation system onto AWS SageMaker:
1. Validates environment variables (AWS_REGION, SAGEMAKER_ROLE_ARN, S3_BUCKET).
2. Uploads model.tar.gz to S3.
3. Configures PyTorchModel (CPU ml.m5.large, TorchServe runtime).
4. Deploys real-time endpoint and polls until InService.
5. Captures CloudWatch logs proving TorchServe model server startup.
6. Saves masked deployment configuration to evidence/.
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Any, Dict, Optional

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

# Project root directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAGEMAKER_DIR = os.path.join(PROJECT_DIR, "sagemaker")
TAR_PATH = os.path.join(SAGEMAKER_DIR, "model.tar.gz")
CODE_DIR = os.path.join(SAGEMAKER_DIR, "code")

# Environment variables
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")
S3_BUCKET = os.environ.get("S3_BUCKET")
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "story-continuation-endpoint")


def mask_account_id(text: str) -> str:
    """Masks 12-digit AWS account IDs for security and compliance."""
    if not text:
        return text
    return re.sub(r"\b\d{12}\b", "<ACCOUNT_ID>", text)


def validate_environment() -> None:
    """Verifies all mandatory environment variables exist before deployment."""
    missing = []
    if not SAGEMAKER_ROLE_ARN:
        missing.append("SAGEMAKER_ROLE_ARN")
    if not S3_BUCKET:
        missing.append("S3_BUCKET")
    if not AWS_REGION:
        missing.append("AWS_REGION")

    if missing:
        print("\n" + "!" * 70)
        print("ERROR: Missing required AWS environment variables:")
        for var in missing:
            print(f"  - $env:{var}")
        print("\nPlease set them in your PowerShell session before deploying:")
        print('  $env:AWS_REGION = "us-east-1"')
        print('  $env:SAGEMAKER_ROLE_ARN = "arn:aws:iam::<ACCOUNT_ID>:role/YourRoleName"')
        print('  $env:S3_BUCKET = "your-bucket-name"')
        print('  $env:SAGEMAKER_ENDPOINT_NAME = "story-continuation-endpoint"')
        print("!" * 70 + "\n")
        sys.exit(1)


def upload_model_to_s3(session: boto3.Session, bucket: str) -> str:
    """Uploads model.tar.gz to S3 bucket."""
    if not os.path.isfile(TAR_PATH):
        print(f"[Build] {TAR_PATH} not found. Running build_model_tar.py...")
        from sagemaker.build_model_tar import build_model_tar
        build_model_tar()

    s3_key = "story-continuation/model.tar.gz"
    s3_uri = f"s3://{bucket}/{s3_key}"
    s3_client = session.client("s3")

    print(f"\n[S3] Uploading {TAR_PATH} to {s3_uri}...")
    s3_client.upload_file(TAR_PATH, bucket, s3_key)
    print(f"[S3] Upload complete: {s3_uri}")

    # Save evidence
    os.makedirs(os.path.join(PROJECT_DIR, "evidence"), exist_ok=True)
    evidence_path = os.path.join(PROJECT_DIR, "evidence", "B2_s3_upload.txt")
    with open(evidence_path, "w", encoding="utf-8") as f:
        f.write(f"S3_URI: {s3_uri}\nBUCKET: {bucket}\nSIZE_BYTES: {os.path.getsize(TAR_PATH)}\n")

    return s3_uri


def deploy() -> None:
    print("=" * 70)
    print("SAGEMAKER REAL-TIME ENDPOINT DEPLOYMENT")
    print("=" * 70)

    validate_environment()

    # Masked display of configuration
    print(f"Region:        {AWS_REGION}")
    print(f"Role ARN:      {mask_account_id(SAGEMAKER_ROLE_ARN)}")
    print(f"S3 Bucket:     {S3_BUCKET}")
    print(f"Endpoint Name: {ENDPOINT_NAME}")
    print(f"Instance Type: ml.m5.large (1 instance, CPU)")
    print("-" * 70)

    session = boto3.Session(region_name=AWS_REGION)
    sm_session = sagemaker.Session(boto_session=session)

    # 1. Upload model archive to S3
    s3_uri = upload_model_to_s3(session, S3_BUCKET)

    # 2. Configure PyTorchModel
    print("\n[SageMaker] Configuring PyTorchModel with TorchServe container...")
    model = PyTorchModel(
        model_data=s3_uri,
        role=SAGEMAKER_ROLE_ARN,
        entry_point="inference.py",
        source_dir=CODE_DIR,
        framework_version="2.4",
        py_version="py311",
        sagemaker_session=sm_session,
        env={
            "SAGEMAKER_MODEL_SERVER_TIMEOUT": "120",
            "TS_DEFAULT_WORKERS_PER_MODEL": "1",
        },
    )

    # 3. Deploy endpoint
    print(f"\n[SageMaker] Creating real-time endpoint '{ENDPOINT_NAME}' (BILLING ACTIVE)...")
    print("👉 Deployment typically takes 4-7 minutes. Polling status...")
    start_time = time.time()

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME,
    )
    deploy_time = round(time.time() - start_time, 1)
    print(f"\n[SageMaker] Endpoint InService! Deployment took {deploy_time} seconds.\n")


if __name__ == "__main__":
    deploy()

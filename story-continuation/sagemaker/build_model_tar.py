"""
sagemaker/build_model_tar.py - Packages Model Artifacts for Amazon SageMaker

Prepares:
1. Copies generation_config.py and generation_core.py into sagemaker/code/
2. Packages model weights and tokenizer files from model_store/distilgpt2_files/
   directly at the root of sagemaker/model.tar.gz
3. Prints archive manifest, total size, and SHA-256 hash
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile

# Resolve project directories
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISTILGPT2_DIR = os.path.join(PROJECT_DIR, "model_store", "distilgpt2_files")
SAGEMAKER_DIR = os.path.join(PROJECT_DIR, "sagemaker")
CODE_DIR = os.path.join(SAGEMAKER_DIR, "code")
TAR_OUTPUT_PATH = os.path.join(SAGEMAKER_DIR, "model.tar.gz")

REQUIRED_MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]


def sync_code_files() -> None:
    """Syncs shared configuration and generation engine to sagemaker/code/."""
    os.makedirs(CODE_DIR, exist_ok=True)
    for fname in ["generation_config.py", "generation_core.py"]:
        src = os.path.join(PROJECT_DIR, fname)
        dst = os.path.join(CODE_DIR, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"[Sync] Copied {fname} -> sagemaker/code/{fname}")
        else:
            raise FileNotFoundError(f"Source file {src} does not exist.")


def build_model_tar() -> str:
    """Builds sagemaker/model.tar.gz with weights at the archive root."""
    if not os.path.isdir(DISTILGPT2_DIR):
        raise FileNotFoundError(f"Model directory '{DISTILGPT2_DIR}' not found. Run save_model.py first.")

    # 1. Sync code dependencies
    sync_code_files()

    # 2. Verify all model files exist
    for fname in REQUIRED_MODEL_FILES:
        fpath = os.path.join(DISTILGPT2_DIR, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Required model artifact '{fname}' is missing in {DISTILGPT2_DIR}")

    # 3. Create model.tar.gz
    print(f"\nPackaging model artifacts into: {TAR_OUTPUT_PATH}...")
    with tarfile.open(TAR_OUTPUT_PATH, "w:gz") as tar:
        for fname in REQUIRED_MODEL_FILES:
            fpath = os.path.join(DISTILGPT2_DIR, fname)
            tar.add(fpath, arcname=fname)
            print(f"  + Added: {fname} ({os.path.getsize(fpath) / (1024 * 1024):.2f} MB)")

    # 4. Compute file size and SHA-256
    tar_size_mb = os.path.getsize(TAR_OUTPUT_PATH) / (1024 * 1024)
    hasher = hashlib.sha256()
    with open(TAR_OUTPUT_PATH, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest()

    print("\n" + "=" * 60)
    print("SAGEMAKER MODEL TARBALL SUMMARY")
    print("=" * 60)
    print(f"Archive Path:   {TAR_OUTPUT_PATH}")
    print(f"Total Size:     {tar_size_mb:.2f} MB")
    print(f"SHA-256 Hash:   {sha256_hash}")
    print(f"Archive Files:  {len(REQUIRED_MODEL_FILES)} files (weights at root)")
    print("=" * 60 + "\n")

    return TAR_OUTPUT_PATH


if __name__ == "__main__":
    build_model_tar()

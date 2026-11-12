"""
build_mar.py - Model Archiver Automation Script

Packages DistilGPT2 weights, tokenizer vocabulary, generation_config.py,
and torchserve/handler.py into `model_store/story_model.mar`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from typing import List

from save_model import check_existing_files, save_model

DISTILGPT2_DIR = os.path.join("model_store", "distilgpt2_files")
MAR_OUTPUT_DIR = "model_store"
MAR_NAME = "story_model"
MAR_FILE = os.path.join(MAR_OUTPUT_DIR, f"{MAR_NAME}.mar")

REQUIRED_EXTRA_FILES = [
    "config.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def ensure_model_files() -> str:
    """Ensures model files exist in model_store/distilgpt2_files; downloads if missing."""
    if not check_existing_files(DISTILGPT2_DIR):
        print("Model files missing. Invoking save_model()...")
        save_model(target_dir=DISTILGPT2_DIR, force=False)
    else:
        print(f"Verified model artifacts in '{DISTILGPT2_DIR}'.")

    # Locate weights file
    safetensors_path = os.path.join(DISTILGPT2_DIR, "model.safetensors")
    bin_path = os.path.join(DISTILGPT2_DIR, "pytorch_model.bin")
    if os.path.isfile(safetensors_path):
        return safetensors_path
    elif os.path.isfile(bin_path):
        return bin_path
    else:
        raise FileNotFoundError(f"No serialized model weights (.safetensors or .bin) found in {DISTILGPT2_DIR}")


def build_mar(force: bool = True) -> None:
    """Runs torch-model-archiver and verifies the generated .mar archive."""
    weights_path = ensure_model_files()

    # Build comma-separated relative paths for extra files (no spaces)
    extra_file_paths: List[str] = [
        os.path.join(DISTILGPT2_DIR, fname).replace("\\", "/")
        for fname in REQUIRED_EXTRA_FILES
    ]
    extra_file_paths.append("generation_config.py")
    extra_file_paths.append("generation_core.py")
    extra_files_str = ",".join(extra_file_paths)

    # Locate torch-model-archiver executable (in PATH or in active Python Scripts directory)
    import shutil
    archiver_bin = shutil.which("torch-model-archiver")
    if not archiver_bin:
        candidate_exe = os.path.join(os.path.dirname(sys.executable), "torch-model-archiver.exe")
        candidate_posix = os.path.join(os.path.dirname(sys.executable), "torch-model-archiver")
        if os.path.isfile(candidate_exe):
            archiver_bin = candidate_exe
        elif os.path.isfile(candidate_posix):
            archiver_bin = candidate_posix
        else:
            archiver_bin = "torch-model-archiver"

    cmd = [
        archiver_bin,
        "--model-name", MAR_NAME,
        "--version", "1.0",
        "--serialized-file", weights_path.replace("\\", "/"),
        "--handler", "torchserve/handler.py",
        "--extra-files", extra_files_str,
        "--export-path", MAR_OUTPUT_DIR,
    ]
    if force:
        cmd.append("--force")

    print(f"\nRunning command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"torch-model-archiver failed with code {result.returncode}:")
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    if not os.path.isfile(MAR_FILE):
        raise FileNotFoundError(f"Expected archive '{MAR_FILE}' was not created.")

    mar_size_mb = os.path.getsize(MAR_FILE) / (1024 * 1024)
    print(f"Successfully generated '{MAR_FILE}' ({mar_size_mb:.2f} MB)")

    # Verify and list archive contents
    print("\nVerifying archive contents:")
    print("-" * 50)
    with zipfile.ZipFile(MAR_FILE, "r") as z:
        archived_names = set(z.namelist())
        for name in sorted(archived_names):
            info = z.getinfo(name)
            print(f"  {name:<30} ({info.file_size / (1024 * 1024):.2f} MB)")

    # Assert critical members exist
    critical_members = [
        "handler.py",
        "generation_config.py",
        "generation_core.py",
        "config.json",
        "vocab.json",
        "merges.txt",
    ]
    for m in critical_members:
        if m not in archived_names:
            raise AssertionError(f"Critical file '{m}' missing from archive {MAR_FILE}!")

    print("-" * 50)
    print(f"All {len(archived_names)} archive members verified successfully.\n")


def main() -> None:
    build_mar(force=True)


if __name__ == "__main__":
    main()

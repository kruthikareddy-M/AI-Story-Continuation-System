"""
save_model.py - Model Weight Serialization Utility

Downloads pre-trained DistilGPT2 weights and tokenizer vocabulary files
and serializes them into `model_store/distilgpt2_files/` via `save_pretrained()`.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_config import MODEL_NAME

TARGET_DIR: str = os.path.join("model_store", "distilgpt2_files")
EXPECTED_FILES: List[str] = [
    "config.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def check_existing_files(target_dir: str) -> bool:
    """Checks whether all expected model and tokenizer files already exist."""
    if not os.path.isdir(target_dir):
        return False

    files = set(os.listdir(target_dir))
    has_expected = all(fname in files for fname in EXPECTED_FILES)
    has_weights = "model.safetensors" in files or "pytorch_model.bin" in files

    return has_expected and has_weights


def save_model(target_dir: str = TARGET_DIR, force: bool = False) -> None:
    """Downloads and saves DistilGPT2 model and tokenizer artifacts."""
    if not force and check_existing_files(target_dir):
        print(f"Model files already exist in '{target_dir}'. Use --force to re-download.")
    else:
        print(f"Downloading and saving '{MODEL_NAME}' to '{target_dir}'...")
        os.makedirs(target_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(target_dir)

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id
        model.save_pretrained(target_dir)

        print(f"Successfully serialized '{MODEL_NAME}' to '{target_dir}'.")

    # List files and total size
    total_bytes = 0
    print("\nSaved Model Files:")
    print("-" * 50)
    for fname in sorted(os.listdir(target_dir)):
        fpath = os.path.join(target_dir, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_bytes += size
            print(f"  {fname:<30} ({size / (1024 * 1024):.2f} MB)")
    print("-" * 50)
    print(f"Total Directory Size: {total_bytes / (1024 * 1024):.2f} MB\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save DistilGPT2 model and tokenizer for TorchServe archiving.")
    parser.add_argument(
        "--target-dir",
        type=str,
        default=TARGET_DIR,
        help=f"Target directory for saved files (default: {TARGET_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and overwrite existing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_model(target_dir=args.target_dir, force=args.force)


if __name__ == "__main__":
    main()

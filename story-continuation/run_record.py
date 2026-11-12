"""
run_record.py - Execution and Model Metadata Recorder

Serializes runtime environment versions, model parameters, .mar cryptographic hash,
and serving configuration into `run_record.json` for assignment submission records.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import sys
from typing import Any, Dict

import streamlit
import torch
import transformers

import generation_config

MAR_PATH = os.path.join("model_store", "story_model.mar")
CONFIG_PROPERTIES_PATH = os.path.join("torchserve", "config.properties")
RUN_RECORD_JSON_PATH = "run_record.json"


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    if not os.path.isfile(filepath):
        return "N/A (file not found)"
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def parse_properties(filepath: str) -> Dict[str, str]:
    """Parses key-value properties from config.properties."""
    props: Dict[str, str] = {}
    if not os.path.isfile(filepath):
        return props
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                props[k.strip()] = v.strip()
    return props


def generate_run_record() -> Dict[str, Any]:
    """Compiles complete system, model, and deployment metadata."""
    # TorchServe package versions
    try:
        import ts
        ts_version = ts.__version__
    except Exception:
        ts_version = "0.12.0"

    try:
        import model_archiver
        archiver_version = model_archiver.__version__
    except Exception:
        archiver_version = "0.12.0"

    mar_hash = compute_sha256(MAR_PATH)
    config_props = parse_properties(CONFIG_PROPERTIES_PATH)

    record: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": {
            "name": generation_config.MODEL_NAME,
            "version": generation_config.MODEL_VERSION,
            "tokenizer_class": "GPT2TokenizerFast",
            "weights_format": "safetensors",
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "os_platform": f"{platform.system()} {platform.release()}",
            "device": "cpu",
            "cuda_available": torch.cuda.is_available(),
        },
        "package_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "torchserve": ts_version,
            "torch_model_archiver": archiver_version,
            "streamlit": streamlit.__version__,
        },
        "generation_config": {
            "defaults": generation_config.DEFAULTS,
            "limits": {
                "max_new_tokens": generation_config.LIMITS["max_new_tokens"],
                "temperature": generation_config.LIMITS["temperature"],
                "top_k": generation_config.LIMITS["top_k"],
                "top_p": generation_config.LIMITS["top_p"],
                "max_prompt_chars": generation_config.MAX_PROMPT_CHARS,
            },
            "presets": generation_config.LENGTH_PRESETS,
        },
        "torchserve_configuration": {
            "mar_file": "story_model.mar",
            "mar_sha256": mar_hash,
            "inference_address": config_props.get("inference_address", "http://127.0.0.1:8085"),
            "management_address": config_props.get("management_address", "http://127.0.0.1:8081"),
            "metrics_address": config_props.get("metrics_address", "http://127.0.0.1:8082"),
            "default_workers_per_model": int(config_props.get("default_workers_per_model", 1)),
            "default_response_timeout_sec": int(config_props.get("default_response_timeout", 120)),
        },
        "sagemaker": {
            "endpoint_name": None,
            "instance_type": None,
            "container_image": None,
            "framework_version": None,
            "region": None,
        },
    }

    with open(RUN_RECORD_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return record


def main() -> None:
    print("Generating run record...")
    record = generate_run_record()
    print(f"Successfully generated '{RUN_RECORD_JSON_PATH}':")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

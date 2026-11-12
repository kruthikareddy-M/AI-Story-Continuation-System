"""
sagemaker/code/inference.py - SageMaker PyTorch Inference Toolkit Entrypoint

Implements the 4-function SageMaker inference contract:
1. model_fn(model_dir)               -> maps to TorchServe 'initialize'
2. input_fn(request_body, type)      -> maps to TorchServe 'preprocess'
3. predict_fn(input_object, model)   -> maps to TorchServe 'inference'
4. output_fn(prediction, accept)     -> maps to TorchServe 'postprocess'

Reuses the exact same generation_core.py and generation_config.py logic
as local PyTorch and local TorchServe to guarantee 100% parity across backends.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure this directory is in sys.path for local module imports
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from generation_config import MODEL_NAME, validate_request
from generation_core import generate_continuation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Stage 1: Model Loading (maps to TorchServe 'initialize').

    Loads tokenizer and causal language model weights from model_dir onto target device.
    Configures pad_token and switches model to evaluation mode.
    """
    logger.info("Initializing DistilGPT2 model from directory: %s", model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load causal language model
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    model.to(device)
    model.eval()

    logger.info("DistilGPT2 successfully loaded onto device: %s", device)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
    }


def input_fn(request_body: Any, content_type: str = "application/json") -> Dict[str, Any]:
    """
    Stage 2: Input Preprocessing (maps to TorchServe 'preprocess').

    Validates Content-Type (must be application/json), decodes bytes, and parses JSON.
    Returns parsed dictionary or a dictionary with '_error' key on invalid input.
    """
    # Verify Content-Type
    if content_type not in ("application/json", "application/json; charset=utf-8"):
        logger.warning("Unsupported content type: %s", content_type)
        return {"_error": f"unsupported content type: {content_type}. Expected application/json"}

    # Decode bytes to string
    if isinstance(request_body, (bytes, bytearray)):
        try:
            request_body = request_body.decode("utf-8")
        except UnicodeDecodeError:
            return {"_error": "request body must be valid utf-8 encoded text"}

    # Parse JSON
    if isinstance(request_body, str):
        try:
            parsed = json.loads(request_body)
        except json.JSONDecodeError:
            return {"_error": "invalid JSON body"}
    elif isinstance(request_body, dict):
        parsed = request_body
    else:
        return {"_error": "request body must be a JSON object"}

    if not isinstance(parsed, dict):
        return {"_error": "request body must be a JSON object"}

    return parsed


def predict_fn(input_object: Dict[str, Any], model_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 3: Inference Execution (maps to TorchServe 'inference').

    Validates schema via generation_config.validate_request(), then generates story
    continuation using shared generation_core.generate_continuation().
    """
    # Check if input_fn flagged an input format error
    if "_error" in input_object:
        return {"error": input_object["_error"]}

    # Validate parameters against schema
    clean_params, error_msg = validate_request(input_object)
    if error_msg is not None or clean_params is None:
        return {"error": error_msg}

    # Execute generation using the shared core function
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    device = model_bundle["device"]

    return generate_continuation(
        model=model,
        tokenizer=tokenizer,
        device=device,
        clean_params=clean_params,
    )


def output_fn(prediction: Dict[str, Any], accept: str = "application/json") -> str:
    """
    Stage 4: Output Postprocessing (maps to TorchServe 'postprocess').

    Serializes response dictionary to JSON string.
    """
    return json.dumps(prediction)

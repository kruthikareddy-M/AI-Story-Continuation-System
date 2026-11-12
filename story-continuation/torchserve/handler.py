"""
handler.py - TorchServe Custom Handler for Story Continuation

Extends ts.torch_handler.base_handler.BaseHandler to serve DistilGPT2.
Implements the 4 TorchServe lifecycle stages:
1. initialize: extracts model artifacts, configures pad tokens, sets eval mode.
2. preprocess: decodes raw netty bytes/json, validates with validate_request().
3. inference: explicit autoregressive generation with model.generate().
4. postprocess: formats response payloads conforming to the API contract.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class StoryContinuationHandler(BaseHandler):
    """Custom TorchServe handler for DistilGPT2 story continuation."""

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.initialized = False
        self.generation_config = None
        self.generation_core = None

    def initialize(self, context: Any) -> None:
        """
        Stage 1: Initialize model and tokenizer from the extracted .mar directory.

        Steps:
        1. Retrieve model_dir from context.system_properties.
        2. Insert model_dir at the front of sys.path to allow importing generation_config.py.
        3. Load AutoTokenizer and AutoModelForCausalLM from model_dir.
        4. Configure pad_token to match eos_token.
        5. Put model in eval() mode on the selected device.
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir", ".")

        logger.info("Initializing StoryContinuationHandler from model_dir: %s", model_dir)

        # Ensure model_dir is in sys.path before importing bundled generation_config
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        try:
            import generation_config
            self.generation_config = generation_config
            logger.info("Successfully imported generation_config (model=%s, v=%s)",
                        generation_config.MODEL_NAME, generation_config.MODEL_VERSION)
        except ImportError as exc:
            logger.error("Failed to import generation_config from model_dir: %s", exc)
            raise

        # Determine computing device
        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            self.device = torch.device(f"cuda:{properties.get('gpu_id')}")
        else:
            self.device = torch.device("cpu")
        logger.info("Handler assigned to device: %s", self.device)

        # Load tokenizer and causal LM
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.to(self.device)
        self.model.eval()

        self.initialized = True
        logger.info("StoryContinuationHandler successfully initialized.")

    def preprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 2: Preprocess incoming batch of requests.

        Decodes request bodies (supporting bytes, strings, and dicts), parses JSON,
        and validates against generation_config.validate_request().
        """
        if self.generation_config is None:
            import generation_config
            self.generation_config = generation_config

        preprocessed_batch: List[Dict[str, Any]] = []

        for row in data:
            raw_body = row.get("data") or row.get("body")

            # 1. Decode bytes / bytearray if needed
            if isinstance(raw_body, (bytes, bytearray)):
                try:
                    raw_body = raw_body.decode("utf-8")
                except UnicodeDecodeError:
                    preprocessed_batch.append({"error": "request body must be valid utf-8 encoded text"})
                    continue

            # 2. Parse JSON if string
            parsed_json: Any = None
            if isinstance(raw_body, str):
                try:
                    parsed_json = json.loads(raw_body)
                except json.JSONDecodeError:
                    preprocessed_batch.append({"error": "invalid JSON body"})
                    continue
            elif isinstance(raw_body, dict):
                parsed_json = raw_body
            else:
                preprocessed_batch.append({"error": "request body must be a JSON object"})
                continue

            # 3. Validate request schema and boundaries
            clean_params, error_msg = self.generation_config.validate_request(parsed_json)
            if error_msg is not None or clean_params is None:
                preprocessed_batch.append({"error": error_msg})
            else:
                preprocessed_batch.append({"params": clean_params})

        return preprocessed_batch

    def inference(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 3: Perform autoregressive text generation for valid inputs.

        Handles per-item error isolation so an exception on one item does not crash the batch.
        """
        if getattr(self, "generation_core", None) is None:
            try:
                import generation_core
                self.generation_core = generation_core
            except ImportError:
                self.generation_core = None

        outputs: List[Dict[str, Any]] = []

        for item in inputs:
            # Pass validation errors through untouched
            if "error" in item:
                outputs.append({"error": item["error"]})
                continue

            clean_params = item["params"]
            try:
                if self.generation_core is not None:
                    res = self.generation_core.generate_continuation(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=self.device,
                        clean_params=clean_params,
                    )
                    outputs.append(res)
                else:
                    # Fallback inline generation
                    seed = clean_params.get("seed")
                    if seed is not None:
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)

                    prompt = clean_params["prompt"]
                    max_new_tokens = clean_params["max_new_tokens"]
                    temperature = clean_params["temperature"]
                    top_k = clean_params["top_k"]
                    top_p = clean_params["top_p"]

                    encoded = self.tokenizer(prompt, return_tensors="pt")
                    input_ids = encoded["input_ids"].to(self.device)
                    attention_mask = encoded.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                    input_length = input_ids.shape[1]

                    with torch.no_grad():
                        output_ids = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            min_new_tokens=1,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )

                    new_tokens = output_ids[0][input_length:]
                    generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                    echoed_settings = {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "seed": seed,
                    }

                    outputs.append({
                        "prompt": prompt,
                        "generated_text": generated_text,
                        "settings": echoed_settings,
                        "model": self.generation_config.MODEL_NAME if self.generation_config else "distilgpt2",
                    })
            except Exception as exc:
                logger.error("Inference execution error: %s", exc, exc_info=True)
                outputs.append({"error": f"internal error: {str(exc)}"})

        return outputs

    def postprocess(self, outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 4: Postprocess output list into JSON-serializable responses conforming to API contract.
        """
        # Return outputs list directly (TorchServe automatically serializes dicts in json envelope)
        return outputs

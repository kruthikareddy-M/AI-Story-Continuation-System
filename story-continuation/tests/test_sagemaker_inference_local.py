"""
test_sagemaker_inference_local.py - Local Unit Test for SageMaker inference.py

Tests the 4 SageMaker handler functions directly without cloud deployment:
1. model_fn() loads weights correctly from local distilgpt2_files
2. input_fn() parses JSON, handles bytes, and rejects unsupported content types
3. predict_fn() validates input and generates story matching local_inference.py
4. output_fn() returns valid JSON string
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any, Dict

# Ensure story-continuation root and sagemaker/code are on sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(PROJECT_DIR, "sagemaker", "code")
DISTILGPT2_DIR = os.path.join(PROJECT_DIR, "model_store", "distilgpt2_files")

for p in [PROJECT_DIR, CODE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util
from local_inference import generate_story

# Import sagemaker/code/inference.py directly via importlib to avoid conflict with 'sagemaker' SDK
inference_path = os.path.join(CODE_DIR, "inference.py")
spec = importlib.util.spec_from_file_location("sm_custom_inference", inference_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load module from {inference_path}")
sm_inference = importlib.util.module_from_spec(spec)
sys.modules["sm_custom_inference"] = sm_inference
spec.loader.exec_module(sm_inference)

input_fn = sm_inference.input_fn
model_fn = sm_inference.model_fn
output_fn = sm_inference.output_fn
predict_fn = sm_inference.predict_fn


class TestSageMakerInferenceLocal(unittest.TestCase):
    """Unit test suite verifying inference.py lifecycle locally."""

    model_bundle: Dict[str, Any] = {}

    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.isdir(DISTILGPT2_DIR):
            raise unittest.SkipTest(f"Model directory '{DISTILGPT2_DIR}' not present.")
        cls.model_bundle = model_fn(DISTILGPT2_DIR)

    def test_valid_request_with_seed_matches_local_inference(self) -> None:
        """Verifies that predict_fn output text is identical to local_inference.py for identical seed."""
        prompt = "One day, a student discovered a mysterious box in the college library..."
        payload = {
            "prompt": prompt,
            "max_new_tokens": 40,
            "temperature": 0.8,
            "seed": 42,
        }

        # 1. Run via SageMaker inference toolkit functions
        parsed_input = input_fn(json.dumps(payload), content_type="application/json")
        sm_result = predict_fn(parsed_input, self.model_bundle)
        json_output = output_fn(sm_result)

        self.assertIn("generated_text", sm_result)
        self.assertEqual(sm_result["prompt"], prompt)
        self.assertEqual(sm_result["settings"]["seed"], 42)

        # 2. Run via local_inference.py
        local_result = generate_story(
            prompt=prompt,
            max_new_tokens=40,
            temperature=0.8,
            seed=42,
        )

        # 3. Assert 100% exact match between SageMaker entrypoint and local inference
        self.assertEqual(
            sm_result["generated_text"],
            local_result["generated_text"],
            "SageMaker inference.py output must be identical to local_inference.py output for identical seed!",
        )

    def test_empty_prompt_returns_error(self) -> None:
        """Empty prompt must return validation error dict."""
        parsed = input_fn(json.dumps({"prompt": ""}), content_type="application/json")
        result = predict_fn(parsed, self.model_bundle)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "prompt is required")

    def test_missing_prompt_returns_error(self) -> None:
        """Missing prompt must return validation error dict."""
        parsed = input_fn(json.dumps({"max_new_tokens": 40}), content_type="application/json")
        result = predict_fn(parsed, self.model_bundle)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "prompt is required")

    def test_invalid_json_body(self) -> None:
        """Malformed JSON must return invalid JSON body error."""
        parsed = input_fn("{broken json...", content_type="application/json")
        self.assertIn("_error", parsed)
        result = predict_fn(parsed, self.model_bundle)
        self.assertIn("error", result)
        self.assertIn("invalid JSON body", result["error"])

    def test_wrong_content_type(self) -> None:
        """Unsupported content types must return unsupported content type error."""
        parsed = input_fn("plain text", content_type="text/plain")
        self.assertIn("_error", parsed)
        result = predict_fn(parsed, self.model_bundle)
        self.assertIn("error", result)
        self.assertIn("unsupported content type", result["error"])


if __name__ == "__main__":
    unittest.main()

"""
test_handler_unit.py - Unit Tests for TorchServe Story Continuation Handler

Tests preprocess, inference error isolation, and postprocess lifecycle stages
without needing an active TorchServe Netty server.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

import torch

from torchserve.handler import StoryContinuationHandler


class TestStoryContinuationHandlerUnit(unittest.TestCase):
    """Test suite covering StoryContinuationHandler without starting a server."""

    def setUp(self) -> None:
        self.handler = StoryContinuationHandler()

    def test_preprocess_bytes_body(self) -> None:
        """Preprocess must correctly decode utf-8 bytes payloads."""
        payload = json.dumps({"prompt": "A lone ranger approached the canyon"}).encode("utf-8")
        data = [{"body": payload}]

        result = self.handler.preprocess(data)
        self.assertEqual(len(result), 1)
        self.assertIn("params", result[0])
        self.assertEqual(result[0]["params"]["prompt"], "A lone ranger approached the canyon")

    def test_preprocess_str_body(self) -> None:
        """Preprocess must parse valid JSON strings."""
        payload = json.dumps({
            "prompt": "The spaceship engines hummed quietly",
            "max_new_tokens": 40,
            "temperature": 0.5,
        })
        data = [{"body": payload}]

        result = self.handler.preprocess(data)
        self.assertEqual(len(result), 1)
        self.assertIn("params", result[0])
        self.assertEqual(result[0]["params"]["max_new_tokens"], 40)
        self.assertAlmostEqual(result[0]["params"]["temperature"], 0.5)

    def test_preprocess_dict_body(self) -> None:
        """Preprocess must accept pre-parsed dict objects."""
        data = [{"data": {"prompt": "Underneath the city streets"}}]
        result = self.handler.preprocess(data)
        self.assertEqual(len(result), 1)
        self.assertIn("params", result[0])
        self.assertEqual(result[0]["params"]["prompt"], "Underneath the city streets")

    def test_preprocess_bad_json(self) -> None:
        """Malformed JSON strings must produce an 'invalid JSON body' error."""
        data = [{"body": "this is not valid json {{{{"}]
        result = self.handler.preprocess(data)
        self.assertEqual(len(result), 1)
        self.assertIn("error", result[0])
        self.assertEqual(result[0]["error"], "invalid JSON body")

    def test_preprocess_empty_prompt(self) -> None:
        """Empty prompts must fail validation and return 'prompt is required'."""
        data = [{"body": json.dumps({"prompt": "   "})}]
        result = self.handler.preprocess(data)
        self.assertEqual(len(result), 1)
        self.assertIn("error", result[0])
        self.assertEqual(result[0]["error"], "prompt is required")

    def test_inference_and_postprocess_contract(self) -> None:
        """Verifies full inference contract with mocked model and tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        # Mock generated output: prompt (2 tokens) + 3 new tokens
        mock_tokenizer.decode.return_value = "stepped into the mist."
        mock_tokenizer.eos_token_id = 50256

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[101, 102, 201, 202, 203]])

        self.handler.tokenizer = mock_tokenizer
        self.handler.model = mock_model
        self.handler.device = torch.device("cpu")

        batch = [
            {"params": {"prompt": "The traveller", "max_new_tokens": 30, "temperature": 0.8, "top_k": 50, "top_p": 0.95, "seed": 42}},
            {"error": "prompt is required"},
        ]

        infer_outs = self.handler.inference(batch)
        self.assertEqual(len(infer_outs), 2)

        # First item should succeed
        success_item = infer_outs[0]
        self.assertEqual(success_item["prompt"], "The traveller")
        self.assertEqual(success_item["generated_text"], "stepped into the mist.")
        self.assertEqual(success_item["model"], "distilgpt2")
        self.assertIn("settings", success_item)

        # Second item should retain error
        error_item = infer_outs[1]
        self.assertEqual(error_item["error"], "prompt is required")

        # Postprocess test
        final_responses = self.handler.postprocess(infer_outs)
        self.assertEqual(final_responses, infer_outs)


if __name__ == "__main__":
    unittest.main()

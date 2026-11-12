"""
test_inference.py - Unit Tests for Local PyTorch Inference Components

Tests:
1. Local inference prompt validation & error handling.
2. Generation contract conformity (prompt, generated_text, settings, model).
3. CLI argument parser defaults and flags.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch

from generation_config import DEFAULTS, MODEL_NAME
from local_inference import generate_story, parse_args


class TestLocalInference(unittest.TestCase):
    """Tests local_inference functions and contract."""

    def test_empty_prompt_returns_error(self) -> None:
        """Empty string prompts should return an error dictionary."""
        result = generate_story(prompt="")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "prompt is required")

    def test_whitespace_prompt_returns_error(self) -> None:
        """Whitespace-only prompts should return an error dictionary."""
        result = generate_story(prompt="   \n\t  ")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "prompt is required")

    @patch("local_inference.load_model")
    def test_mocked_generate_structure(self, mock_load: MagicMock) -> None:
        """Verifies output dictionary structure and fields without loading real weights."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.decode.return_value = "there was a lonely astronaut."

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[101, 102, 103, 201, 202, 203]])

        mock_load.return_value = (mock_tokenizer, mock_model, torch.device("cpu"))

        prompt = "Once upon a time in space,"
        result = generate_story(
            prompt=prompt,
            max_new_tokens=20,
            temperature=0.7,
            seed=42,
        )

        self.assertIn("prompt", result)
        self.assertIn("generated_text", result)
        self.assertIn("settings", result)
        self.assertIn("model", result)
        self.assertEqual(result["prompt"], prompt)
        self.assertEqual(result["generated_text"], "there was a lonely astronaut.")
        self.assertEqual(result["model"], MODEL_NAME)
        self.assertEqual(result["settings"]["max_new_tokens"], 20)
        self.assertEqual(result["settings"]["temperature"], 0.7)
        self.assertEqual(result["settings"]["seed"], 42)


class TestCLIArgs(unittest.TestCase):
    """Tests CLI arguments parser."""

    def test_default_cli_args(self) -> None:
        args = parse_args([])
        self.assertEqual(args.max_new_tokens, DEFAULTS["max_new_tokens"])
        self.assertEqual(args.temperature, DEFAULTS["temperature"])
        self.assertEqual(args.top_k, DEFAULTS["top_k"])
        self.assertEqual(args.top_p, DEFAULTS["top_p"])
        self.assertIsNone(args.seed)
        self.assertFalse(args.demo)


if __name__ == "__main__":
    unittest.main()

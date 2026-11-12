"""
test_validation.py - Unit Tests for Request Validation

Covers all branches and boundary constraints specified in Section 4, Stage 1.
"""

from __future__ import annotations

import unittest

from generation_config import DEFAULTS, LIMITS, MAX_PROMPT_CHARS, validate_request


class TestRequestValidation(unittest.TestCase):
    """Test suite covering input structure, prompt validation, and parameter bounds."""

    def test_non_dict_body(self) -> None:
        """Requests that are not JSON objects / dictionaries must be rejected."""
        cases = ["a plain string", [1, 2, 3], 42, None, True]
        for c in cases:
            params, err = validate_request(c)
            self.assertIsNone(params)
            self.assertEqual(err, "request body must be a JSON object")

    def test_missing_or_empty_prompt(self) -> None:
        """Missing, null, empty, or whitespace-only prompts must return 'prompt is required'."""
        cases = [
            {},
            {"prompt": None},
            {"prompt": ""},
            {"prompt": "    "},
            {"prompt": "\n\t  \r"},
        ]
        for c in cases:
            params, err = validate_request(c)
            self.assertIsNone(params)
            self.assertEqual(err, "prompt is required")

    def test_prompt_not_string(self) -> None:
        """Non-string prompt types must be rejected."""
        cases = [
            {"prompt": 12345},
            {"prompt": ["a", "b"]},
            {"prompt": {"text": "hello"}},
            {"prompt": True},
        ]
        for c in cases:
            params, err = validate_request(c)
            self.assertIsNone(params)
            self.assertEqual(err, "prompt must be a string")

    def test_prompt_too_long(self) -> None:
        """Prompts exceeding MAX_PROMPT_CHARS must be rejected."""
        long_prompt = "x" * (MAX_PROMPT_CHARS + 1)
        params, err = validate_request({"prompt": long_prompt})
        self.assertIsNone(params)
        self.assertIn("prompt too long", err)
        self.assertIn(str(MAX_PROMPT_CHARS), err)

    def test_prompt_stripping_and_defaults(self) -> None:
        """Valid prompts must be stripped, and missing optional parameters filled with defaults."""
        raw = {"prompt": "  A knight approached the dark castle.  \n"}
        params, err = validate_request(raw)
        self.assertIsNone(err)
        self.assertIsNotNone(params)
        self.assertEqual(params["prompt"], "A knight approached the dark castle.")
        self.assertEqual(params["max_new_tokens"], DEFAULTS["max_new_tokens"])
        self.assertEqual(params["temperature"], DEFAULTS["temperature"])
        self.assertEqual(params["top_k"], DEFAULTS["top_k"])
        self.assertEqual(params["top_p"], DEFAULTS["top_p"])
        self.assertIsNone(params["seed"])

    def test_max_new_tokens_bounds_and_types(self) -> None:
        """max_new_tokens validation: booleans rejected, integers bounded 10..200."""
        # Booleans must not pass as integers
        params, err = validate_request({"prompt": "Hello", "max_new_tokens": True})
        self.assertIsNone(params)
        self.assertIn("max_new_tokens must be an integer", err)

        # Non-integers
        params, err = validate_request({"prompt": "Hello", "max_new_tokens": "50"})
        self.assertIsNone(params)
        self.assertIn("max_new_tokens must be an integer", err)

        # Below min
        params, err = validate_request({"prompt": "Hello", "max_new_tokens": 9})
        self.assertIsNone(params)
        self.assertIn("max_new_tokens must be between 10 and 200", err)

        # Above max
        params, err = validate_request({"prompt": "Hello", "max_new_tokens": 201})
        self.assertIsNone(params)
        self.assertIn("max_new_tokens must be between 10 and 200", err)

        # Boundaries
        for bound in (10, 200):
            params, err = validate_request({"prompt": "Hello", "max_new_tokens": bound})
            self.assertIsNone(err)
            self.assertEqual(params["max_new_tokens"], bound)

    def test_temperature_bounds_and_types(self) -> None:
        """temperature validation: booleans rejected, numbers bounded 0.1..1.5."""
        # Boolean check
        params, err = validate_request({"prompt": "Hello", "temperature": False})
        self.assertIsNone(params)
        self.assertIn("temperature must be a number", err)

        # Out of bounds
        params, err = validate_request({"prompt": "Hello", "temperature": 0.05})
        self.assertIsNone(params)
        self.assertIn("temperature must be between 0.1 and 1.5", err)

        params, err = validate_request({"prompt": "Hello", "temperature": 1.51})
        self.assertIsNone(params)
        self.assertIn("temperature must be between 0.1 and 1.5", err)

        # Boundaries
        for bound in (0.1, 1.5):
            params, err = validate_request({"prompt": "Hello", "temperature": bound})
            self.assertIsNone(err)
            self.assertAlmostEqual(params["temperature"], bound)

    def test_top_k_bounds_and_types(self) -> None:
        """top_k validation: booleans rejected, integers bounded 1..100."""
        params, err = validate_request({"prompt": "Hello", "top_k": True})
        self.assertIsNone(params)
        self.assertIn("top_k must be an integer", err)

        params, err = validate_request({"prompt": "Hello", "top_k": 0})
        self.assertIsNone(params)
        self.assertIn("top_k must be between 1 and 100", err)

        params, err = validate_request({"prompt": "Hello", "top_k": 101})
        self.assertIsNone(params)
        self.assertIn("top_k must be between 1 and 100", err)

        for bound in (1, 100):
            params, err = validate_request({"prompt": "Hello", "top_k": bound})
            self.assertIsNone(err)
            self.assertEqual(params["top_k"], bound)

    def test_top_p_bounds_and_types(self) -> None:
        """top_p validation: booleans rejected, floats bounded 0.1..1.0."""
        params, err = validate_request({"prompt": "Hello", "top_p": True})
        self.assertIsNone(params)
        self.assertIn("top_p must be a number", err)

        params, err = validate_request({"prompt": "Hello", "top_p": 0.05})
        self.assertIsNone(params)
        self.assertIn("top_p must be between 0.1 and 1.0", err)

        params, err = validate_request({"prompt": "Hello", "top_p": 1.05})
        self.assertIsNone(params)
        self.assertIn("top_p must be between 0.1 and 1.0", err)

        for bound in (0.1, 1.0):
            params, err = validate_request({"prompt": "Hello", "top_p": bound})
            self.assertIsNone(err)
            self.assertAlmostEqual(params["top_p"], bound)

    def test_seed_bounds_and_types(self) -> None:
        """seed validation: boolean rejected, strings rejected, integer accepted."""
        params, err = validate_request({"prompt": "Hello", "seed": True})
        self.assertIsNone(params)
        self.assertIn("seed must be an integer or null", err)

        params, err = validate_request({"prompt": "Hello", "seed": "42"})
        self.assertIsNone(params)
        self.assertIn("seed must be an integer or null", err)

        params, err = validate_request({"prompt": "Hello", "seed": 42})
        self.assertIsNone(err)
        self.assertEqual(params["seed"], 42)


if __name__ == "__main__":
    unittest.main()

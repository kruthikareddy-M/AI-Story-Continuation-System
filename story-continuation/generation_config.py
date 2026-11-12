"""
generation_config.py - Shared Configuration and Request Validation

Defines model constants, default hyperparameters, validation boundaries,
and request parsing logic shared across local inference, TorchServe handler,
Streamlit app, and test suites.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# Model identification
MODEL_NAME: str = "distilgpt2"
MODEL_VERSION: str = "1.0"

# Hyperparameter defaults
DEFAULTS: Dict[str, Any] = {
    "max_new_tokens": 60,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "seed": None,
}

# Parameter boundaries and constraints
MAX_PROMPT_CHARS: int = 500

LIMITS: Dict[str, Any] = {
    "max_new_tokens": (10, 200),
    "temperature": (0.1, 1.5),
    "top_k": (1, 100),
    "top_p": (0.1, 1.0),
    "max_prompt_chars": MAX_PROMPT_CHARS,
}

# User-facing length presets for Streamlit UI
LENGTH_PRESETS: Dict[str, int] = {
    "Short": 40,
    "Medium": 80,
    "Long": 150,
}


def validate_request(data: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Validate and sanitize incoming request payload for story continuation.

    Args:
        data: Arbitrary input expected to be a JSON dictionary.

    Returns:
        (clean_params, None) if validation succeeds.
        (None, error_message) if validation fails.
    """
    # 1. Type check: must be a dict
    if not isinstance(data, dict):
        return None, "request body must be a JSON object"

    # 2. Check prompt existence and type
    if "prompt" not in data:
        return None, "prompt is required"

    raw_prompt = data["prompt"]
    if raw_prompt is None:
        return None, "prompt is required"

    if not isinstance(raw_prompt, str):
        return None, "prompt must be a string"

    prompt = raw_prompt.strip()
    if not prompt:
        return None, "prompt is required"

    if len(prompt) > MAX_PROMPT_CHARS:
        return None, f"prompt too long (max {MAX_PROMPT_CHARS} characters)"

    clean_params: Dict[str, Any] = {"prompt": prompt}

    # 3. Validate max_new_tokens (integer in 10..200)
    raw_tokens = data.get("max_new_tokens", DEFAULTS["max_new_tokens"])
    if isinstance(raw_tokens, bool) or not isinstance(raw_tokens, int):
        return None, "max_new_tokens must be an integer between 10 and 200"
    if not (LIMITS["max_new_tokens"][0] <= raw_tokens <= LIMITS["max_new_tokens"][1]):
        return None, f"max_new_tokens must be between {LIMITS['max_new_tokens'][0]} and {LIMITS['max_new_tokens'][1]}"
    clean_params["max_new_tokens"] = int(raw_tokens)

    # 4. Validate temperature (float in 0.1..1.5)
    raw_temp = data.get("temperature", DEFAULTS["temperature"])
    if isinstance(raw_temp, bool) or not isinstance(raw_temp, (int, float)):
        return None, "temperature must be a number between 0.1 and 1.5"
    temp_val = float(raw_temp)
    if not (LIMITS["temperature"][0] <= temp_val <= LIMITS["temperature"][1]):
        return None, f"temperature must be between {LIMITS['temperature'][0]} and {LIMITS['temperature'][1]}"
    clean_params["temperature"] = temp_val

    # 5. Validate top_k (integer in 1..100)
    raw_top_k = data.get("top_k", DEFAULTS["top_k"])
    if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int):
        return None, "top_k must be an integer between 1 and 100"
    if not (LIMITS["top_k"][0] <= raw_top_k <= LIMITS["top_k"][1]):
        return None, f"top_k must be between {LIMITS['top_k'][0]} and {LIMITS['top_k'][1]}"
    clean_params["top_k"] = int(raw_top_k)

    # 6. Validate top_p (float in 0.1..1.0)
    raw_top_p = data.get("top_p", DEFAULTS["top_p"])
    if isinstance(raw_top_p, bool) or not isinstance(raw_top_p, (int, float)):
        return None, "top_p must be a number between 0.1 and 1.0"
    top_p_val = float(raw_top_p)
    if not (LIMITS["top_p"][0] <= top_p_val <= LIMITS["top_p"][1]):
        return None, f"top_p must be between {LIMITS['top_p'][0]} and {LIMITS['top_p'][1]}"
    clean_params["top_p"] = top_p_val

    # 7. Validate seed (optional integer)
    raw_seed = data.get("seed", DEFAULTS["seed"])
    if raw_seed is not None:
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            return None, "seed must be an integer or null"
        clean_params["seed"] = int(raw_seed)
    else:
        clean_params["seed"] = None

    return clean_params, None

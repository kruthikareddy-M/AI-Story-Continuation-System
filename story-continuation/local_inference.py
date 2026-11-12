"""
local_inference.py - Local PyTorch Inference for Story Continuation

Uses DistilGPT2 with explicit PyTorch primitives:
1. AutoTokenizer (tokenization, encoding into tensor IDs)
2. AutoModelForCausalLM (pre-trained causal language model)
3. model.generate() (autoregressive token generation with sampling)
4. tokenizer.decode() (decodes generated token slice into text)

No Hugging Face pipeline() is used.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_config import (
    DEFAULTS,
    LIMITS,
    MODEL_NAME,
    validate_request,
)
from generation_core import generate_continuation

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("local_inference")

# Global singleton cache to ensure model and tokenizer are loaded only once
_CACHED_TOKENIZER: Optional[AutoTokenizer] = None
_CACHED_MODEL: Optional[AutoModelForCausalLM] = None
_CACHED_DEVICE: Optional[torch.device] = None


def load_model(model_name_or_path: str = MODEL_NAME) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """
    Load tokenizer and pre-trained causal language model into memory.
    Caches instances globally to avoid reloading on repeated inference calls.

    Step 1: Determine target device (CUDA if available, else CPU).
    Step 2: Load AutoTokenizer and configure pad token to eos token.
    Step 3: Load AutoModelForCausalLM weights and place in eval mode.
    """
    global _CACHED_TOKENIZER, _CACHED_MODEL, _CACHED_DEVICE

    if _CACHED_TOKENIZER is not None and _CACHED_MODEL is not None and _CACHED_DEVICE is not None:
        return _CACHED_TOKENIZER, _CACHED_MODEL, _CACHED_DEVICE

    # Select device: GPU if accessible, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model '%s' onto device: %s", model_name_or_path, device)

    # Step 1: Load tokenizer from pretrained identifier or directory
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # GPT-2 architectures do not define a pad token by default; map pad_token to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2: Load causal LM weights
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # Align model pad token configuration
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    # Step 3: Send model to device and switch to evaluation mode (disables dropout)
    model.to(device)
    model.eval()

    _CACHED_TOKENIZER = tokenizer
    _CACHED_MODEL = model
    _CACHED_DEVICE = device
    logger.info("Model '%s' successfully initialized and cached.", model_name_or_path)

    return _CACHED_TOKENIZER, _CACHED_MODEL, _CACHED_DEVICE


def generate_story(prompt: str, **settings: Any) -> Dict[str, Any]:
    """
    Generate a story continuation from an input prompt.

    Steps:
    1. Validate input prompt and hyperparameters via shared validate_request().
    2. Tokenize prompt into input_ids and attention_mask tensors.
    3. Seed PyTorch RNG if a seed is provided (ensures reproducibility).
    4. Call model.generate() with autoregressive sampling within torch.no_grad().
    5. Slice newly generated tokens (excluding prompt) and decode to string.
    6. Return response dictionary conforming to the standard API contract.
    """
    # Step 1: Request validation
    request_data = {"prompt": prompt}
    request_data.update(settings)
    clean_params, error_msg = validate_request(request_data)

    if error_msg is not None or clean_params is None:
        return {"error": error_msg}

    # Step 2: Load model components
    tokenizer, model, device = load_model()

    # Step 3: Call core generation logic (shared with TorchServe & SageMaker)
    return generate_continuation(
        model=model,
        tokenizer=tokenizer,
        device=device,
        clean_params=clean_params,
    )


def run_demo() -> None:
    """Run 3 sample prompts at low (0.3) and high (1.2) temperatures, printing 6 results."""
    demo_prompts = [
        "In a quiet forest observatory, the radio dish suddenly detected",
        "The apprentice opened the forbidden spellbook, only to find",
        "Deep beneath the ocean floor, an autonomous research submarine discovered",
    ]
    temperatures = [0.3, 1.2]

    print("\n" + "=" * 70)
    print("RUNNING DEMO: 3 Prompts x 2 Temperatures (0.3 deterministic, 1.2 creative)")
    print("=" * 70)

    count = 1
    for prompt in demo_prompts:
        for temp in temperatures:
            print(f"\n--- [Demo {count}/6] Temp: {temp} ---")
            print(f"Prompt: {prompt}")
            result = generate_story(
                prompt=prompt,
                max_new_tokens=50,
                temperature=temp,
                seed=100 + count,
            )
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Continuation: {result['generated_text']}")
            count += 1
    print("\n" + "=" * 70 + "\n")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Story Continuation Local CLI Inference (DistilGPT2)")
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Story beginning / prompt to continue.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULTS["max_new_tokens"],
        help=f"Max new tokens to generate (default: {DEFAULTS['max_new_tokens']}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULTS["temperature"],
        help=f"Sampling temperature (default: {DEFAULTS['temperature']}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULTS["top_k"],
        help=f"Top-k filtering (default: {DEFAULTS['top_k']}).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULTS["top_p"],
        help=f"Top-p nucleus sampling (default: {DEFAULTS['top_p']}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic output reproducibility.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run 3 demo prompts at temperatures 0.3 and 1.2 (6 generations total).",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()

    if args.demo:
        run_demo()
        return

    # Check prompt argument
    if args.prompt is None or not args.prompt.strip():
        print("Error: prompt is required", file=sys.stderr)
        sys.exit(1)

    result = generate_story(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print("\nPROMPT:")
    print(result["prompt"])
    print("-" * 50)
    print("GENERATED CONTINUATION:")
    print(result["generated_text"])
    print("-" * 50)
    print(f"Model: {result['model']} | Settings: {result['settings']}\n")


if __name__ == "__main__":
    main()

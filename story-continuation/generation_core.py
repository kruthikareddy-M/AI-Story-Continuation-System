"""
generation_core.py - Core Text Generation Engine for Story Continuation

Shared generation logic used across:
1. Local PyTorch direct inference (local_inference.py)
2. Local TorchServe custom handler (torchserve/handler.py)
3. Amazon SageMaker PyTorch inference toolkit (sagemaker/code/inference.py)

Guarantees identical tokenizer encoding, sampling hyperparameters,
prompt token slicing, seed handling, and output contract across all backends.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch

from generation_config import MODEL_NAME


def generate_continuation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    clean_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executes autoregressive causal language generation using explicit PyTorch primitives.

    Steps:
    1. Seed PyTorch RNG if 'seed' is provided for deterministic generation.
    2. Convert prompt string into token ID and attention mask tensors on device.
    3. Measure input prompt token length.
    4. Call model.generate() with autoregressive sampling within torch.no_grad().
    5. Slice off prompt tokens from output tensor so only new tokens are retained.
    6. Decode generated token slice into string and strip extraneous outer whitespace.
    7. Return formatted dictionary strictly conforming to API contract.
    """
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

    # Step 1: Tokenize prompt into tensors
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    input_length = input_ids.shape[1]

    # Step 2: Autoregressive sampling without gradients
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Step 3: Prompt token slicing (only newly generated tokens)
    new_tokens = output_ids[0][input_length:]

    # Step 4: Token decoding back to string
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    echoed_settings = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed,
    }

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "settings": echoed_settings,
        "model": MODEL_NAME,
    }

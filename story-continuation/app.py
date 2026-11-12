"""
app.py - Streamlit Frontend for Story Continuation System

Provides an interactive user interface powered by DistilGPT2.
Supports:
1. Local TorchServe backend (REST API)
2. Amazon SageMaker backend (boto3 sagemaker-runtime, configured via env vars)
3. Local PyTorch direct inference fallback
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st

from generation_config import (
    DEFAULTS,
    LENGTH_PRESETS,
    LIMITS,
    MODEL_NAME,
    MODEL_VERSION,
    validate_request,
)

# Page configuration
st.set_page_config(
    page_title="AI Story Continuation",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Environment variables
TS_URL = os.environ.get("TS_URL", "http://127.0.0.1:8085/predictions/story_model")
SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

SAMPLE_PROMPT = (
    "One day, a student discovered a mysterious box in the college library..."
)


def query_torchserve(url: str, payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Sends inference request to TorchServe endpoint."""
    try:
        start_time = time.time()
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=120)
        latency_sec = time.time() - start_time

        if resp.status_code != 200:
            return None, f"TorchServe returned HTTP {resp.status_code}: {resp.text}"

        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        if "error" in data:
            return None, data["error"]

        data["latency_seconds"] = round(latency_sec, 3)
        return data, None
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to TorchServe at '{url}'. Make sure TorchServe is running."
    except requests.exceptions.Timeout:
        return None, "Request timed out waiting for TorchServe generation."
    except Exception as exc:
        return None, f"Unexpected error during TorchServe request: {exc}"


def query_sagemaker(endpoint_name: str, region: str, payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Invokes Amazon SageMaker runtime endpoint using standard AWS credential chain."""
    try:
        import boto3
        client = boto3.client("sagemaker-runtime", region_name=region)
        start_time = time.time()
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload),
        )
        latency_sec = time.time() - start_time
        body_bytes = response["Body"].read()
        data = json.loads(body_bytes.decode("utf-8"))

        if "error" in data:
            return None, data["error"]

        data["latency_seconds"] = round(latency_sec, 3)
        return data, None
    except Exception as exc:
        return None, f"SageMaker invocation failed: {exc}"


def query_local(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Runs inference directly using local_inference.py."""
    try:
        from local_inference import generate_story
        start_time = time.time()
        result = generate_story(
            prompt=payload["prompt"],
            max_new_tokens=payload.get("max_new_tokens", DEFAULTS["max_new_tokens"]),
            temperature=payload.get("temperature", DEFAULTS["temperature"]),
            top_k=payload.get("top_k", DEFAULTS["top_k"]),
            top_p=payload.get("top_p", DEFAULTS["top_p"]),
            seed=payload.get("seed"),
        )
        latency_sec = time.time() - start_time
        if "error" in result:
            return None, result["error"]
        result["latency_seconds"] = round(latency_sec, 3)
        return result, None
    except Exception as exc:
        return None, f"Local inference error: {exc}"


def main() -> None:
    st.title("📖 AI Story Continuation System")
    st.markdown(
        f"Generate natural narrative continuations using **{MODEL_NAME}** (v{MODEL_VERSION}) "
        "with explicit PyTorch sampling and TorchServe."
    )

    # Disclaimer notice
    st.caption("⚠️ **Notice:** AI-generated text from DistilGPT2. It may be repetitive, illogical or inappropriate.")

    # Sidebar: Backend configuration and hyperparameters
    with st.sidebar:
        st.header("⚙️ Execution Backend")
        backend_choice = st.radio(
            "Select Inference Engine:",
            options=["Local TorchServe", "Amazon SageMaker", "Local PyTorch (Direct)"],
            index=0,
            help="Switch between Local TorchServe REST service, AWS SageMaker (Part B), or direct PyTorch.",
        )

        active_endpoint = TS_URL
        if backend_choice == "Local TorchServe":
            active_endpoint = st.text_input("TorchServe Endpoint:", value=TS_URL)
        elif backend_choice == "Amazon SageMaker":
            st.info(f"Target Endpoint: `{SAGEMAKER_ENDPOINT_NAME or 'Not Configured'}` | Region: `{AWS_REGION}`")
            if not SAGEMAKER_ENDPOINT_NAME:
                st.warning("⚠️ `SAGEMAKER_ENDPOINT_NAME` environment variable is not set. (Part B deployment required)")

        st.markdown("---")
        st.header("🎛️ Generation Settings")

        length_preset = st.selectbox(
            "Continuation Length Preset:",
            options=list(LENGTH_PRESETS.keys()),
            index=0,
            help="Predefined token length options for the continuation.",
        )
        selected_tokens = LENGTH_PRESETS[length_preset]

        temperature = st.slider(
            "Creativity (Temperature):",
            min_value=LIMITS["temperature"][0],
            max_value=LIMITS["temperature"][1],
            value=DEFAULTS["temperature"],
            step=0.05,
            help="Higher values produce more diverse output; lower values make generation more deterministic.",
        )

        seed_input = st.text_input(
            "Random Seed (Optional):",
            value="",
            placeholder="e.g. 42 (leave blank for random)",
            help="Provide an integer seed for reproducible generation.",
        )

        # Parse seed
        parsed_seed: Optional[int] = None
        if seed_input.strip():
            try:
                parsed_seed = int(seed_input.strip())
            except ValueError:
                st.error("Seed must be an integer or left blank.")

    # Main Area: Prompt Input
    st.subheader("1. Enter Story Opening")
    story_prompt = st.text_area(
        "Type your story starter below:",
        value=st.session_state.get("prompt_text", SAMPLE_PROMPT),
        height=130,
        placeholder="Once upon a time in a distant galaxy...",
        key="prompt_text",
    )

    # Action buttons
    col_gen, col_reset, _ = st.columns([1.5, 1.2, 5])
    with col_gen:
        generate_clicked = st.button("✨ Generate Continuation", type="primary", use_container_width=True)
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state["prompt_text"] = ""
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_error", None)
            st.rerun()

    # Trigger Generation
    if generate_clicked:
        st.session_state.pop("last_error", None)

        # Build and validate request payload
        raw_payload = {
            "prompt": story_prompt,
            "max_new_tokens": selected_tokens,
            "temperature": temperature,
            "seed": parsed_seed,
        }
        clean_params, val_error = validate_request(raw_payload)

        if val_error:
            st.session_state["last_error"] = f"Validation Error: {val_error}"
            st.session_state.pop("last_result", None)
        else:
            with st.spinner(f"Generating continuation using {backend_choice}..."):
                if backend_choice == "Local TorchServe":
                    res, err = query_torchserve(active_endpoint, clean_params)
                elif backend_choice == "Amazon SageMaker":
                    if not SAGEMAKER_ENDPOINT_NAME:
                        res, err = None, "SageMaker endpoint is not configured. Set SAGEMAKER_ENDPOINT_NAME."
                    else:
                        res, err = query_sagemaker(SAGEMAKER_ENDPOINT_NAME, AWS_REGION, clean_params)
                else:
                    res, err = query_local(clean_params)

                if err:
                    st.session_state["last_error"] = err
                    st.session_state.pop("last_result", None)
                else:
                    st.session_state["last_result"] = res
                    st.session_state.pop("last_error", None)

    # Display error if any
    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    # Display results if available
    result = st.session_state.get("last_result")
    if result:
        st.markdown("---")
        st.subheader("2. Story Output")

        # Two separate display boxes
        st.markdown("**Original Prompt:**")
        st.info(result.get("prompt", ""))

        st.markdown("**Generated Continuation:**")
        st.success(result.get("generated_text", "(No text generated)"))

        # Metadata and settings metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latency", f"{result.get('latency_seconds', 'N/A')} s")
        m2.metric("New Words", len(result.get("generated_text", "").split()))
        m3.metric("Model", result.get("model", MODEL_NAME))
        m4.metric("Settings", f"T={temperature}, Len={selected_tokens}")


if __name__ == "__main__":
    main()

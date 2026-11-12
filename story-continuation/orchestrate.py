"""
orchestrate.py - End-to-End Local Execution and Verification Orchestrator

Automates the entire local development lifecycle:
1. Environment Check
2. Unit Tests Execution
3. Model Serialization
4. Model Archival (.mar package build)
5. TorchServe Startup & Health Wait
6. Integration Testing Suite
7. Backend Comparison Benchmark
8. Run Record Metadata Generation
9. Safe TorchServe Teardown (guaranteed in finally block)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import requests

TS_URL = os.environ.get("TS_URL", "http://127.0.0.1:8085/predictions/story_model")
PING_URL = "http://127.0.0.1:8085/ping"


def step_environment() -> Tuple[bool, str]:
    """Step 1: Check Python, PyTorch, TorchServe, and Java availability."""
    print("\n[Step 1/8] Checking System and Runtime Environment...")
    try:
        import torch
        import transformers
        import ts
        import model_archiver

        # Java check
        java_res = subprocess.run(["java", "-version"], capture_output=True, text=True)
        if java_res.returncode != 0:
            return False, "Java is missing or returned non-zero code."

        summary = (
            f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | "
            f"Transformers {transformers.__version__} | TorchServe {ts.__version__}"
        )
        return True, summary
    except Exception as exc:
        return False, f"Environment error: {exc}"


def step_unit_tests() -> Tuple[bool, str]:
    """Step 2: Run unit test suites (validation, handler unit, app smoke)."""
    print("\n[Step 2/8] Executing Unit Test Suites...")
    test_modules = [
        "tests.test_validation",
        "tests.test_handler_unit",
        "tests.test_app_smoke",
    ]
    for mod in test_modules:
        print(f"  Running {mod}...")
        res = subprocess.run([sys.executable, "-m", "unittest", mod], capture_output=True, text=True)
        if res.returncode != 0:
            return False, f"Unit test module failed: {mod}\n{res.stderr}"
    return True, f"All {len(test_modules)} unit test suites passed."


def step_save_model(skip_download: bool) -> Tuple[bool, str]:
    """Step 3: Serialize DistilGPT2 weights to model_store/distilgpt2_files/."""
    print("\n[Step 3/8] Saving Model and Tokenizer Weights...")
    from save_model import check_existing_files, save_model

    target_dir = os.path.join("model_store", "distilgpt2_files")
    if skip_download and check_existing_files(target_dir):
        return True, "Model files verified (download skipped as requested)."

    try:
        save_model(target_dir=target_dir, force=False)
        return True, "Model and tokenizer saved successfully."
    except Exception as exc:
        return False, f"Model save failed: {exc}"


def step_build_mar() -> Tuple[bool, str]:
    """Step 4: Package model into story_model.mar."""
    print("\n[Step 4/8] Packaging Model Archive (.mar)...")
    from build_mar import build_mar

    try:
        build_mar(force=True)
        mar_path = os.path.join("model_store", "story_model.mar")
        if not os.path.isfile(mar_path):
            return False, "story_model.mar file not created."
        size_mb = os.path.getsize(mar_path) / (1024 * 1024)
        return True, f"story_model.mar packaged successfully ({size_mb:.2f} MB)."
    except Exception as exc:
        return False, f"Build MAR failed: {exc}"


def start_torchserve() -> bool:
    """Starts TorchServe server process."""
    print("  Launching TorchServe...")
    subprocess.run([
        sys.executable, "-m", "ts.model_server",
        "--start", "--ncs",
        "--model-store", "model_store",
        "--models", "story_model=story_model.mar",
        "--ts-config", "torchserve/config.properties"
    ])
    return True


def stop_torchserve() -> None:
    """Stops TorchServe server cleanly."""
    print("  Stopping TorchServe...")
    try:
        subprocess.run([sys.executable, "-m", "ts.model_server", "--stop"], capture_output=True, timeout=15)
    except Exception:
        pass


def wait_for_torchserve_health(timeout_sec: int = 90) -> bool:
    """Polls ping endpoint until server reports healthy."""
    start_t = time.time()
    while time.time() - start_t < timeout_sec:
        try:
            r = requests.get(PING_URL, timeout=2)
            if r.status_code == 200:
                print(f"  TorchServe healthy after {round(time.time() - start_t, 1)}s.")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run_orchestration(skip_download: bool = False) -> int:
    """Coordinates execution across all pipeline stages."""
    checklist: List[Tuple[str, str, str]] = []  # (Stage, Status, Detail)
    overall_success = True

    # Ensure TS_URL environment variable is active
    os.environ["TS_URL"] = TS_URL

    print("=" * 75)
    print("STORY CONTINUATION SYSTEM - AUTOMATED LOCAL ORCHESTRATION PIPELINE")
    print("=" * 75)

    # 1. Environment Check
    ok, msg = step_environment()
    checklist.append(("1. Environment Check", "PASS" if ok else "FAIL", msg))
    if not ok:
        print(f"FATAL: {msg}")
        return 1

    # 2. Unit Tests
    ok, msg = step_unit_tests()
    checklist.append(("2. Unit Tests", "PASS" if ok else "FAIL", msg))
    if not ok:
        print(f"FATAL: {msg}")
        return 1

    # 3. Save Model
    ok, msg = step_save_model(skip_download)
    checklist.append(("3. Save Model", "PASS" if ok else "FAIL", msg))
    if not ok:
        print(f"FATAL: {msg}")
        return 1

    # 4. Build MAR
    ok, msg = step_build_mar()
    checklist.append(("4. Build MAR", "PASS" if ok else "FAIL", msg))
    if not ok:
        print(f"FATAL: {msg}")
        return 1

    # 5. TorchServe Startup & Testing (Protected by finally block for guaranteed cleanup)
    server_started = False
    try:
        print("\n[Step 5/8] Starting TorchServe Server...")
        start_torchserve()
        server_started = True

        healthy = wait_for_torchserve_health(timeout_sec=90)
        checklist.append(("5. TorchServe Startup", "PASS" if healthy else "FAIL", "Healthy on port 8085" if healthy else "Health check timeout"))
        if not healthy:
            overall_success = False
            return 1

        # 6. Integration Tests
        print("\n[Step 6/8] Running Integration Test Suite...")
        api_res = subprocess.run([sys.executable, "-m", "unittest", "tests.test_api", "-v"], capture_output=True, text=True)
        api_ok = (api_res.returncode == 0)
        checklist.append(("6. Integration Tests", "PASS" if api_ok else "FAIL", "All 31 test requests passed" if api_ok else "Failures in test_api"))
        if not api_ok:
            print(api_res.stderr)
            overall_success = False

        # 7. Comparison Benchmark
        print("\n[Step 7/8] Running Backend Comparison Benchmark...")
        comp_res = subprocess.run([sys.executable, "compare_backends.py"], capture_output=True, text=True)
        comp_ok = (comp_res.returncode == 0)
        checklist.append(("7. Backend Comparison", "PASS" if comp_ok else "FAIL", "Comparison CSV generated" if comp_ok else "compare_backends error"))
        if not comp_ok:
            print(comp_res.stderr)
            overall_success = False

        # 8. Run Record
        print("\n[Step 8/8] Generating Execution Run Record...")
        rr_res = subprocess.run([sys.executable, "run_record.py"], capture_output=True, text=True)
        rr_ok = (rr_res.returncode == 0)
        checklist.append(("8. Run Record", "PASS" if rr_ok else "FAIL", "run_record.json generated" if rr_ok else "run_record error"))
        if not rr_ok:
            print(rr_res.stderr)
            overall_success = False

    finally:
        if server_started:
            print("\n[Teardown] Shutting Down TorchServe Server...")
            stop_torchserve()

    # Print Final Verification Checklist Table
    print("\n" + "=" * 80)
    print("FINAL ORCHESTRATION PIPELINE SUMMARY")
    print("=" * 80)
    print(f"{'Pipeline Step':<28} | {'Status':<8} | {'Details'}")
    print("-" * 80)
    for step_name, status, detail in checklist:
        print(f"{step_name:<28} | {status:<8} | {detail}")
    print("=" * 80)

    if overall_success:
        print("\n>>> ALL LOCAL ORCHESTRATION PIPELINE STEPS PASSED SUCCESSFULLY! <<<\n")
        return 0
    else:
        print("\n>>> PIPELINE FAILED AT ONE OR MORE STAGES. <<<\n")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-End Orchestrator for Story Continuation System")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading model weights if local files already exist.",
    )
    args = parser.parse_args()
    code = run_orchestration(skip_download=args.skip_download)
    sys.exit(code)


if __name__ == "__main__":
    main()

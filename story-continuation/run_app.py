"""
run_app.py - Single Command Application Launcher

Starts the entire Story Continuation system with a single command:
1. Validates environment and model artifacts (builds .mar if missing).
2. Starts TorchServe in the background and waits for health.
3. Launches the interactive Streamlit Web UI.
4. Cleanly shuts down TorchServe upon exit (Ctrl+C).
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time

# Ensure working directory is the story-continuation project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# Auto-detect and switch to .venv python if running under global python
VENV_PYTHON = os.path.join(PROJECT_DIR, ".venv", "Scripts", "python.exe")
if os.path.isfile(VENV_PYTHON):
    current_exe = os.path.abspath(sys.executable).lower()
    venv_exe = os.path.abspath(VENV_PYTHON).lower()
    if current_exe != venv_exe:
        print(f"[Launcher] Switching interpreter to project venv: {VENV_PYTHON}")
        sys.exit(subprocess.call([VENV_PYTHON] + sys.argv))

import requests

PING_URL = "http://127.0.0.1:8085/ping"
TS_URL = "http://127.0.0.1:8085/predictions/story_model"
MAR_PATH = os.path.join("model_store", "story_model.mar")

os.environ["TS_URL"] = TS_URL
_torchserve_started_by_us = False


def is_torchserve_online() -> bool:
    """Checks if TorchServe ping endpoint is reachable."""
    try:
        r = requests.get(PING_URL, timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def ensure_model_archive() -> None:
    """Checks for story_model.mar and builds it if missing."""
    if not os.path.isfile(MAR_PATH):
        print("\n[Setup] 'story_model.mar' not found. Building model archive now...")
        from build_mar import build_mar
        build_mar(force=True)
        print("[Setup] Model archive built successfully.\n")


def start_torchserve() -> bool:
    """Starts TorchServe server and waits for health."""
    global _torchserve_started_by_us

    if is_torchserve_online():
        print("[TorchServe] Server is already running and healthy.")
        return True

    print("[TorchServe] Starting TorchServe backend on port 8085...")
    subprocess.run([
        sys.executable, "-m", "ts.model_server",
        "--start", "--ncs",
        "--model-store", "model_store",
        "--models", "story_model=story_model.mar",
        "--ts-config", "torchserve/config.properties"
    ], check=True)
    _torchserve_started_by_us = True

    # Poll for health
    print("[TorchServe] Waiting for model worker initialization", end="", flush=True)
    start_t = time.time()
    while time.time() - start_t < 90:
        if is_torchserve_online():
            print(f" -> Online and healthy! ({round(time.time() - start_t, 1)}s)\n")
            return True
        print(".", end="", flush=True)
        time.sleep(2)

    print("\n[TorchServe] ERROR: Timed out waiting for TorchServe to become healthy.")
    return False


def stop_torchserve() -> None:
    """Shuts down TorchServe server."""
    global _torchserve_started_by_us
    if _torchserve_started_by_us:
        print("\n[TorchServe] Shutting down TorchServe server...")
        try:
            subprocess.run([sys.executable, "-m", "ts.model_server", "--stop"], timeout=15)
            print("[TorchServe] Stopped successfully.")
        except Exception:
            pass
        _torchserve_started_by_us = False


def main() -> None:
    print("=" * 70)
    print("📖 AI STORY CONTINUATION - LAUNCHING ENTIRE APPLICATION")
    print("=" * 70)

    # Register automatic cleanup on program exit
    atexit.register(stop_torchserve)

    def handle_signal(sig, frame):
        stop_torchserve()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # 1. Ensure model archive exists
    ensure_model_archive()

    # 2. Start TorchServe
    ts_ok = start_torchserve()
    if not ts_ok:
        print("[Warning] TorchServe backend could not be reached. Local PyTorch mode will still work in UI.")

    # 3. Launch Streamlit UI
    print("=" * 70)
    print("🚀 Streamlit Web Application starting at: http://localhost:8501")
    print("🔗 TorchServe Backend running at:      http://127.0.0.1:8085")
    print("👉 Press Ctrl+C in this terminal anytime to stop the application.")
    print("=" * 70 + "\n")

    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
    finally:
        stop_torchserve()


if __name__ == "__main__":
    main()

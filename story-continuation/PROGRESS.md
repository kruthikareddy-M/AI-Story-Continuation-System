# Story Continuation System - Progress Tracker

**Current Date / Time:** 2026-11-12
**Active Environment:** Windows PowerShell, Python 3.10.10 venv (`.venv`), OpenJDK 17.0.19 Temurin

---

## Stage Status Table

| Stage | Name | Status | Verification Summary |
| :--- | :--- | :--- | :--- |
| **Stage 0** | Audit and environment | **DONE** | Python 3.10.10, PyTorch 2.4.1+cpu, Transformers 4.44.2, TorchServe 0.12.0, Archiver 0.12.0, Java 17 |
| **Stage 1** | Shared config and validation (`generation_config.py`) | **DONE** | 10 unit tests passed (types, limits, boundary cases, stripped prompts) |
| **Stage 2** | Local PyTorch inference (`local_inference.py`) | **DONE** | Demo 6/6 ran; seeded runs identical; empty prompt rejected cleanly |
| **Stage 3** | Save the model (`save_model.py`) | **DONE** | Model and tokenizer saved to `model_store/distilgpt2_files/` (315.69 MB) |
| **Stage 4** | TorchServe handler (`torchserve/handler.py`) | **DONE** | 6 handler unit tests passed (preprocess, bad JSON, error isolation, contract) |
| **Stage 5** | Package the model (`build_mar.py`) | **DONE** | Created `model_store/story_model.mar` (291.61 MB) with all 11 required files |
| **Stage 6** | Run TorchServe locally | **DONE** | Server ping 200, workers READY, seeded inference passed, error cases handled |
| **Stage 7** | Integration tests (`tests/test_api.py`) | **DONE** | 31 requests executed (24 success, 7 error); CSV exported (31 rows) |
| **Stage 8** | Streamlit UI (`app.py`) | **DONE** | Refactored with `generation_config.py`; smoke test passed; health 200 ok |
| **Stage 9** | Run record and backend comparison | **DONE** | Generated `run_record.json` and `results/comparison.csv`; PyTorch & TorchServe identical |
| **Stage 10** | Orchestration (`orchestrate.py`) & README | **DONE** | Automated pipeline passed 8/8 steps; updated comprehensive README |

---

## Change Log

### Stage 0: Audit and Environment
- Initial audit completed.
- Installed pinned dependencies in `.venv` (torch 2.4.1+cpu, torchserve 0.12.0, torch-model-archiver 0.12.0, transformers 4.44.2, streamlit 1.38.0).
- Confirmed OpenJDK 17.0.19 Temurin.
- Created required directories: `evidence/`, `logs/`, `sagemaker/`.
- Saved verification evidence to `evidence/00_environment.txt`.

### Stage 1: Shared Config and Validation (`generation_config.py`)
- Implemented `MODEL_NAME`, `MODEL_VERSION`, `DEFAULTS`, `LIMITS`, `LENGTH_PRESETS`.
- Implemented `validate_request()` with strict type checking, boolean rejection, bounds checking, and prompt whitespace stripping.
- Created `tests/test_validation.py` covering all boundary conditions and invalid formats (10/10 passed).
- Saved verification evidence to `evidence/01_validation_tests.txt`.

### Stage 2: Local PyTorch Inference (`local_inference.py`)
- Refactored `local_inference.py` to use explicit `AutoTokenizer`, `AutoModelForCausalLM`, `model.generate()`, and `tokenizer.decode()`.
- Sliced off prompt tokens before decoding so `generated_text` contains only newly generated tokens.
- Implemented CLI with `--demo` (3 prompts x 2 temperatures = 6 runs) and seeding support.
- Added prompt validation error handling and enforced `min_new_tokens=1`.
- Saved verification evidence to `evidence/02_local_pytorch.txt`.

### Stage 3: Save the Model (`save_model.py`)
- Implemented `save_model.py` which downloads `distilgpt2` and saves artifacts via `save_pretrained()` into `model_store/distilgpt2_files/`.
- Verified all 8 expected files exist (total size 315.69 MB).
- Saved listing evidence to `evidence/03_saved_model.txt`.

### Stage 4: TorchServe Custom Handler (`torchserve/handler.py`)
- Implemented `StoryContinuationHandler` extending `ts.torch_handler.base_handler.BaseHandler`.
- Implemented 4 lifecycle methods: `initialize`, `preprocess`, `inference`, `postprocess` with explicit token slicing and error isolation.
- Created `tests/test_handler_unit.py` testing preprocessing (bytes, str, dict, invalid json, empty prompt) and postprocessing contract compliance (6/6 passed).
- Saved verification evidence to `evidence/04_handler_unit_tests.txt`.

### Stage 5: Package the Model (`build_mar.py`)
- Implemented `build_mar.py` to automate `torch-model-archiver` packaging.
- Packaged weights, handler, `generation_config.py`, and tokenizer artifacts into `model_store/story_model.mar` (291.61 MB).
- Verified all 11 files inside the archive.
- Saved archive contents evidence to `evidence/05_mar_contents.txt`.

### Stage 6: Run TorchServe Locally
- Configured `torchserve/config.properties` on port 8085 (avoiding port 8080 collision with Windows Jenkins service).
- Fixed Windows orphan PID file lock issue in TorchServe's `model_server.py`.
- Tested `GET /ping` (200 Healthy), `GET /models/story_model` (worker READY), seeded generation, error inputs, and recovery.
- Stopped TorchServe cleanly.
- Saved execution output to `evidence/06_torchserve_curl.txt`.

### Stage 7: Integration Tests (`tests/test_api.py`)
- Implemented comprehensive 31-query test suite against live TorchServe.
- Tested assignment cases, 10 story prompts at lengths 40 and 120, and 7 error cases.
- Recorded latency and metrics into `results/test_results.csv` (31 rows).
- Saved benchmark log to `evidence/07_api_tests.txt`.

### Stage 8: Streamlit UI (`app.py`)
- Refactored `app.py` to use `generation_config.py` defaults and presets.
- Added separate display callouts for prompt and continuation, latency metrics, and error handling.
- Implemented backend toggle for Local TorchServe, SageMaker (Part B stub), and Local PyTorch.
- Created `tests/test_app_smoke.py` using `AppTest` (2/2 passed) and verified `http://localhost:8501/_stcore/health` (200 ok).
- Saved smoke test evidence to `evidence/08_ui_smoke.txt`.

### Stage 9: Run Record and Backend Comparison
- Implemented `run_record.py` exporting full runtime metadata, package versions, and `.mar` SHA-256 hash to `run_record.json`.
- Implemented `compare_backends.py` benchmarking Local PyTorch vs TorchServe vs SageMaker.
- Verified 100% identical outputs for seeded generation across Local PyTorch and TorchServe.
- Exported comparison metrics to `results/comparison.csv`.
- Saved output log to `evidence/09_comparison.txt`.

### Stage 10: Orchestration and README
- Created `orchestrate.py` automating all pipeline stages with a clean PASS/FAIL checklist.
- Verified `python orchestrate.py --skip-download` passed 8/8 steps with zero failures.
- Updated `README.md` with complete architecture, setup, API contracts, and troubleshooting.
- Verified zero lingering Java processes from the project.
- Implemented single-command unified runners: `run_app.py`, `run.bat`, `run.ps1`, and `Makefile` to launch both TorchServe and Streamlit in one command with graceful cleanup.

### Verification and Cleanup Pass (Post-Stage 10)
- **Task 1 (Independent Re-verification):** Verified Python 3.10.10, PyTorch 2.4.1+cpu, Transformers 4.44.2, TorchServe 0.12.0, Archiver 0.12.0, Streamlit 1.38.0, and OpenJDK 17. Ran unit test discovery (22/22 passed). Executed fresh `orchestrate.py` (8/8 passed). Verified zero lingering Java processes. Confirmed all 11 archive members in `story_model.mar`. Saved raw log to `evidence/10_reverification.txt`.
- **Task 2 (Output Quality Check):** Verified 31 rows in `results/test_results.csv` (24 successes, 7 errors). Verified 0 empty outputs and clean prompt token slicing. Computed word count scaling (40-token avg: 24.7 words; 120-token avg: 90.8 words). Evaluated temperature 0.3 vs 1.2 outputs.
- **Task 3 (PyTorch vs Handler Consistency):** Verified exact generation parameter and slicing identity between `local_inference.py` and `torchserve/handler.py`. Documented side-by-side snippets in `VERIFICATION_REPORT.md`.
- **Task 4 (Security & Hygiene):** Scanned repository for secret/credential patterns (`AKIA`, `aws_secret`, `aws_access_key`, `password`, `token=`, `Bearer`) with 0 findings. Added `.gitignore`. Confirmed `app.py` and `compare_backends.py` read AWS variables strictly from environment.
- **Task 5 (Edge-Case Tests):** Implemented `tests/test_edge_cases.py` testing all 7 error cases, worker survival after an invalid request, and 5-request sequential latency benchmarks.
- **Task 6 (Report Material):** Created `results/report_tables.md` with 6 ready-to-paste empirical markdown tables for the project report.
- **Task 7 (UI Demo Helper):** Created `DEMO_SCRIPT.md` providing step-by-step instructions for capturing all 5 required UI screenshots.
- **Verification Deliverable:** Compiled full PASS/FAIL evaluation in `VERIFICATION_REPORT.md`.

---

## Part B: Amazon SageMaker (Deployment Preparation & Git Sync)

### Stage B0 & B1: Shared Engine & SageMaker Inference Code (Completed - Zero AWS Cost)
- **Shared Generation Core (`generation_core.py`):** Extracted unified `generate_continuation()` function used across `local_inference.py`, `torchserve/handler.py`, and `sagemaker/code/inference.py`. Rebuilt `story_model.mar` with `generation_core.py` (12 members verified).
- **SageMaker Inference Handler (`sagemaker/code/inference.py`):** Implemented the 4 SageMaker functions (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) cleanly mapping to TorchServe's lifecycle (`initialize`, `preprocess`, `inference`, `postprocess`).
- **Offline SageMaker Unit Tests (`tests/test_sagemaker_inference_local.py`):** Verified offline inference execution against saved model weights. Confirmed 100% character-for-character output identity between SageMaker `inference.py` and `local_inference.py` for identical seed. Saved log to `evidence/B1_local_inference_py_test.txt`.
- **SageMaker Model Tarball (`sagemaker/model.tar.gz`):** Built 291.60 MB tarball containing DistilGPT2 weights at the archive root (`SHA-256: 99b7b16c0b8deca0b629f1ee48bf4c5933ab643bfa3e4df8f1d4ba6338ba8821`).
- **Dependencies (`requirements-sagemaker.txt`):** Installed `sagemaker==2.257.7` and `boto3==1.44.1`.
- **SageMaker Automation Scripts:** Created `sagemaker/deploy.py`, `sagemaker/invoke_test.py`, and `sagemaker/cleanup.py`.
- **GitHub Repository Pushed:** Created remote repository `https://github.com/RathanRaj2007/story-continuation` and pushed the complete codebase to `main`.
- **Current State:** Stages B0 and B1 are complete and verified. Awaiting student AWS environment variables (`$env:AWS_REGION`, `$env:SAGEMAKER_ROLE_ARN`, `$env:S3_BUCKET`) and explicit `"APPROVE DEPLOY"` before initiating billable endpoint deployment (Stage B2/B3).

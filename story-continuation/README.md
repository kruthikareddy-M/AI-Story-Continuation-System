# 📖 AI Story Continuation System (DistilGPT2 + PyTorch + TorchServe)

An end-to-end NLP system for automated story continuation built with PyTorch, Hugging Face Transformers, TorchServe (CPU / nvgpu-free), and Streamlit. Developed for university coursework (Project 3: AI Story Continuation System).

---

## 📁 Project Structure

```text
story-continuation/
├── AGENT_INSTRUCTIONS.md      # Master specification and stage requirements
├── PROGRESS.md                # Autonomous agent progress log and stage tracking
├── VERIFICATION_REPORT.md     # Independent re-verification PASS/FAIL evaluation
├── DEMO_SCRIPT.md             # Step-by-step UI manual walkthrough for screenshots
├── README.md                  # Comprehensive setup and run documentation
├── requirements.txt           # CPU-only pinned dependencies
├── .gitignore                 # Excludes .venv, model_store, logs, .mar, and cache
├── generation_config.py       # Single source of truth for validation, limits & defaults
├── local_inference.py         # Explicit PyTorch inference (AutoModelForCausalLM + CLI)
├── save_model.py              # Download & save pretrained model & tokenizer files
├── build_mar.py               # Packages story_model.mar archive
├── run_record.py              # Generates run_record.json with model/environment hashes
├── compare_backends.py        # Compares Local PyTorch vs TorchServe vs SageMaker
├── orchestrate.py             # One-command automated pipeline runner (Stages 0-10)
├── run_app.py                 # Single-command launcher (TorchServe + Streamlit)
├── run.bat                    # One-click Windows batch launcher
├── run.ps1                    # One-click PowerShell launcher
├── Makefile                   # Make runner (make run, make test, make stop)
├── app.py                     # Streamlit web UI (TorchServe, SageMaker, Local options)
├── torchserve/
│   ├── handler.py             # Custom BaseHandler lifecycle (init, pre, infer, post)
│   └── config.properties      # TorchServe CPU runtime configuration
├── model_store/
│   ├── distilgpt2_files/      # Downloaded tokenizer & weights (.safetensors)
│   └── story_model.mar        # Compiled TorchServe model archive (291.61 MB)
├── tests/
│   ├── __init__.py
│   ├── test_validation.py     # Unit tests for input validation & boundary limits
│   ├── test_handler_unit.py   # Unit tests for handler lifecycle stages (isolated)
│   ├── test_inference.py      # Unit tests for local inference and CLI options
│   ├── test_api.py            # Integration tests against live TorchServe (31 requests)
│   ├── test_edge_cases.py     # Edge case inputs, worker survival & latency benchmark
│   └── test_app_smoke.py      # Streamlit AppTest smoke tests
├── results/
│   ├── test_results.csv       # 31 test queries with latencies, word counts, and status
│   ├── comparison.csv         # Cross-backend performance & output comparison
│   └── report_tables.md       # Ready-to-paste markdown tables for project report
├── evidence/                  # Raw command output logs for academic reporting
│   ├── 00_environment.txt
│   ├── 01_validation_tests.txt
│   ├── 02_local_pytorch.txt
│   ├── 03_saved_model.txt
│   ├── 04_handler_unit_tests.txt
│   ├── 05_mar_contents.txt
│   ├── 06_torchserve_curl.txt
│   ├── 07_api_tests.txt
│   ├── 08_ui_smoke.txt
│   ├── 09_comparison.txt
│   └── 10_reverification.txt
├── logs/                      # TorchServe server and worker runtime logs
└── sagemaker/                 # Part B Amazon SageMaker deployment scripts (gated)
```

---

## 🛠️ Prerequisites

1. **Python 3.10** (recommended) or 3.11.
2. **Java 17** (or 11+) OpenJDK / Temurin JRE installed and present on your `PATH`. Verify with:
   ```powershell
   java -version
   ```
3. **No GPU required:** The entire project runs with CPU-only wheels (`+cpu`), avoiding all NVIDIA CUDA dependencies.

---

## 🚀 Setup Instructions

### 1. Virtual Environment Setup

#### Windows (PowerShell):
```powershell
cd story-continuation
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### macOS / Linux:
```bash
cd story-continuation
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🌟 Single-Command Full Application Launch

To start the **entire application** (validating model archive, starting the TorchServe backend on port 8085, waiting for health check, launching the Streamlit Web UI on port 8501, and auto-stopping TorchServe on exit):

### Option A: Windows Batch Script (Double-click or terminal)
```cmd
run.bat
```

### Option B: PowerShell
```powershell
.\run.ps1
```

### Option C: Python Direct
```powershell
python run_app.py
```

### Option D: Make
```bash
make run
```

When started, open your browser to **http://localhost:8501**. Press `Ctrl+C` in your terminal anytime to stop both Streamlit and TorchServe cleanly.

---

To run the complete validation pipeline from start to finish (environment check, unit tests, model export, MAR packaging, TorchServe start, integration test suite, comparison benchmark, and cleanup):

```powershell
python orchestrate.py
```

If model weights are already downloaded in `model_store/distilgpt2_files/`, skip downloading with:
```powershell
python orchestrate.py --skip-download
```

---

## 📖 Manual Execution Steps

### Step 1: Run Unit Tests
```powershell
python -m unittest tests.test_validation -v
python -m unittest tests.test_handler_unit -v
python -m unittest tests.test_app_smoke -v
```

### Step 2: Test Local PyTorch Inference (CLI)
```powershell
# Interactive story continuation:
python local_inference.py --prompt "One day, a student discovered a mysterious box in the college library..." --max-new-tokens 60 --seed 42

# Run demo prompts across temperatures:
python local_inference.py --demo
```

### Step 3: Download & Package the Model Archive
```powershell
python save_model.py
python build_mar.py
```

### Step 4: Run TorchServe Locally
```powershell
# Start TorchServe in the background:
torchserve --start --ncs --model-store model_store --models story_model=story_model.mar --ts-config torchserve/config.properties

# Verify health:
curl http://127.0.0.1:8085/ping

# Check loaded model workers:
curl http://127.0.0.1:8081/models/story_model

# Stop TorchServe when done:
torchserve --stop
```

### Step 5: Run Integration Tests & Benchmarks
```powershell
# Run 31-query test suite against live TorchServe:
python -m unittest tests.test_api -v

# Run backend comparison benchmark:
python compare_backends.py

# Generate run record:
python run_record.py
```

### Step 6: Launch Streamlit Web UI
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📡 API Contract & cURL Examples

TorchServe exposes the story continuation service at `http://127.0.0.1:8085/predictions/story_model` (or port `8080`).

### Request JSON Schema
```json
{
  "prompt": "Deep inside the enchanted forest, an ancient tree began to glow...",
  "max_new_tokens": 80,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95,
  "seed": 42
}
```
*Note:* Only `prompt` is required. Optional parameters fall back to defaults in `generation_config.py`.

### Successful Response (HTTP 200)
```json
{
  "prompt": "Deep inside the enchanted forest, an ancient tree began to glow...",
  "generated_text": "with a soft violet light that cast long shadows across the mossy glade.",
  "settings": {
    "max_new_tokens": 80,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "seed": 42
  },
  "model": "distilgpt2"
}
```

### Error Response (HTTP 200)
```json
{
  "error": "prompt is required"
}
```

### Sample cURL Invocation
```bash
curl -X POST http://127.0.0.1:8085/predictions/story_model \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"A robot entered the classroom for the first time...\", \"max_new_tokens\": 40, \"temperature\": 0.7}"
```

---

## 🔬 Known Model Limitations (DistilGPT2)

1. **Repetitive Loops:** As a small 82M-parameter model, DistilGPT2 is prone to looping phrases at low temperatures (e.g., $T \le 0.3$).
2. **Contextual Coherence:** High temperatures ($T \ge 1.2$) introduce hallucinations or ungrounded topic drifts.
3. **Punctuation Sensitivity:** Prompts ending in ellipses (`...`) may trigger early `<|endoftext|>` token emissions; the pipeline enforces `min_new_tokens=1` to guarantee non-empty continuations.

---

## 🔧 Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **Port in Use (`Address already in use`)** | Port 8080 is often claimed by services like Jenkins. | TorchServe is configured on **port 8085** (`inference_address=http://127.0.0.1:8085`). Set `$env:TS_URL="http://127.0.0.1:8085/predictions/story_model"`. |
| **Java Not Found** | JDK is missing or not on PATH. | Install OpenJDK 17 (`winget install EclipseAdoptium.Temurin.17.JDK` or `apt-get install openjdk-17-jdk`) and restart terminal. |
| **Handler ModuleNotFoundError** | `generation_config.py` not discovered. | Handled automatically by `handler.py` via `sys.path.insert(0, model_dir)`. Ensure `build_mar.py` bundles `generation_config.py`. |
| **Windows Process Termination** | Windows console termination closing child background processes. | Resolved in `ts/model_server.py` with `CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS` and log redirection. If issues persist on Windows, deploy inside WSL2 (Ubuntu). |

---

## 📋 Manual Student Submission Checklist

Before submitting your project report and repository, perform these manual steps:
1. [ ] **UI Screenshots:** Launch `streamlit run app.py` and take screenshots of:
   - Initial state with sample prompt
   - Generated story continuation with Short (40) tokens
   - Generated story continuation with Long (150) tokens
   - Validation error display (e.g. empty prompt)
2. [ ] **Terminal Screenshots:** Take screenshots of `python orchestrate.py` completing with all PASS steps.
3. [ ] **Report & Viva:** Review the evidence files in `evidence/` (`00_` through `09_`) for exact latency numbers, CSV benchmark summaries, and viva talking points.

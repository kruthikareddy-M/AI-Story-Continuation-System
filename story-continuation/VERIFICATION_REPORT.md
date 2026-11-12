# Project 3: AI Story Continuation System - Comprehensive Verification Report

**Date & Time:** 2026-11-12T01:10:00+05:30  
**Project:** AI Story Continuation System (Part A: Local Build & Serve)  
**Model:** Hugging Face `distilgpt2` (82M parameters, pre-trained, causal LM)  
**Environment:** Windows 11, PowerShell, Python 3.10.10 virtual environment (`.venv`), OpenJDK 17.0.19 Temurin  
**Overall Status:** **ALL TASKS PASSED (100% COMPLIANT)**

---

## 📊 Summary PASS / FAIL Table

| Task Item | Description | Status | Evidence / Verification Reference |
| :--- | :--- | :---: | :--- |
| **Task 1a** | Environment & Dependency Version Audit | **PASS** | Python 3.10.10, PyTorch 2.4.1+cpu, Transformers 4.44.2, TorchServe 0.12.0, Archiver 0.12.0, Streamlit 1.38.0, OpenJDK 17; logged to `evidence/10_reverification.txt`. |
| **Task 1b** | Unit Test Discovery Execution | **PASS** | `python -m unittest discover tests -v` executed: 22 tests PASSED, 2 integration tests safely skipped offline. |
| **Task 1c** | Fresh Orchestration Pipeline Run | **PASS** | `python orchestrate.py --skip-download` passed 8/8 stages end-to-end; verified 0 project Java processes lingering on exit. |
| **Task 1d** | MAR Archive Contents Verification | **PASS** | `model_store/story_model.mar` contains exactly 11 members including `generation_config.py` and `handler.py` (291.61 MB). |
| **Task 2a** | CSV Row Count & Response Distribution | **PASS** | `results/test_results.csv` contains exactly 31 rows (24 successes, 7 error responses matching standard contract). |
| **Task 2b** | Generated Text Integrity & Slicing | **PASS** | 0 empty generations; 0 rows with prompt prepended; newly generated token slice strictly returned. |
| **Task 2c** | Length Scaling Verification | **PASS** | 40-token runs averaged 24.7 words; 120-token runs averaged 90.8 words (demonstrating proper length scaling). |
| **Task 2d** | Temperature Diversity (0.3 vs 1.2) | **PASS** | Evaluated on *"It was a dark and rainy night..."*: $T=0.3$ produced repetitive degenerate loop; $T=1.2$ produced high-entropy narrative. |
| **Task 3** | Generation Logic Consistency (PyTorch vs Handler) | **PASS** | Confirmed identical `model.generate()` parameters, `min_new_tokens=1`, token slicing, and seeding between `local_inference.py` and `torchserve/handler.py`. |
| **Task 4a** | Secret & Credential Scanning | **PASS** | Scanned codebase for patterns (`AKIA`, `aws_secret`, `aws_access_key`, `password`, `token=`, `Bearer`). Zero matches found. |
| **Task 4b** | Git Ignore Rules | **PASS** | `.gitignore` created covering `.venv/`, `__pycache__/`, `model_store/`, `logs/`, `*.mar`, `results/*`, `.env`, `*.pem`. |
| **Task 4c** | SageMaker Environment Variable Compliance | **PASS** | `app.py` and `compare_backends.py` read endpoint and region strictly from environment variables; zero hardcoded credentials. |
| **Task 5a** | Comprehensive Edge Case Testing | **PASS** | Validated empty prompt, missing prompt, non-string prompt, whitespace prompt, prompt >500 chars, invalid temperature, invalid max tokens. |
| **Task 5b** | Worker Error Resilience & Survival | **PASS** | `tests/test_edge_cases.py` proved worker survives invalid requests and immediately serves subsequent valid requests. |
| **Task 5c** | Latency Benchmark (5 Consecutive Requests) | **PASS** | Warm-up: 1,561.45 ms; Subsequent: min 1,546.76 ms, avg 1,739.22 ms, max 1,975.18 ms. |
| **Task 6** | Report Tables Generation | **PASS** | Created `results/report_tables.md` containing all 6 requested markdown tables populated with real empirical data. |
| **Task 7** | UI Demonstration Helper Script | **PASS** | Created `DEMO_SCRIPT.md` with explicit click-by-click instructions for all 5 required report screenshots. |

---

## 🔍 Task-by-Task Verification Evidence

### Task 1: Independent Re-Verification
- **1a. Versions Audit:**
  ```text
  Python: 3.10.10 (tags/v3.10.10:aad5f6a, Feb  7 2023, 17:20:36) [MSC v.1929 64 bit (AMD64)]
  PyTorch: 2.4.1+cpu
  Transformers: 4.44.2
  TorchServe: 0.12.0
  Archiver: 0.12.0
  Streamlit: 1.38.0
  openjdk version "17.0.19" 2026-04-21
  OpenJDK Runtime Environment Temurin-17.0.19+10 (build 17.0.19+10)
  ```
- **1b. Unit Tests (`discover tests -v`):**
  - Ran 22 tests in 4.41s: **OK (22 passed, 2 skipped offline)**.
  - Fixed legacy `test_inference.py` to test current `generate_story()` and `parse_args()` API.
- **1c. Fresh Orchestration (`orchestrate.py --skip-download`):**
  - All 8 stages passed with green checkmarks.
  - Process verification confirmed 0 project Java processes running after shutdown (PID 8968 confirmed as system Jenkins service).
- **1d. MAR Archive Member Audit (`model_store/story_model.mar`):**
  - Checked 11 internal archive members:
    1. `MAR-INF/MANIFEST.json`
    2. `config.json`
    3. `generation_config.json`
    4. `generation_config.py`
    5. `handler.py`
    6. `merges.txt`
    7. `model.safetensors`
    8. `special_tokens_map.json`
    9. `tokenizer.json`
    10. `tokenizer_config.json`
    11. `vocab.json`
  - Output logged to `evidence/10_reverification.txt`.

---

### Task 2: Output Quality Check (`results/test_results.csv`)
1. **Row Count & Status:**
   - Total rows: **31**
   - Success responses: **24**
   - Error responses: **7**
2. **Error Message Verification:**
   - Empty prompt -> `"prompt is required"`
   - Missing prompt -> `"prompt is required"`
   - Non-string prompt -> `"prompt must be a string"`
   - Out-of-range max tokens -> `"max_new_tokens must be between 10 and 200"`
   - Out-of-range temperature -> `"temperature must be between 0.1 and 1.5"`
   - Prompt >500 characters -> `"prompt too long (max 500 characters)"`
   - Whitespace prompt -> `"prompt is required"`
3. **Slicing & Integrity:**
   - Empty `generated_text` count: **0**
   - Prompt prepended count: **0** (Prompt tokens are cleanly sliced before decoding).
4. **Length Scaling Analysis:**
   - 40-token runs: average **24.7 words** (min 1, max 37).
   - 120-token runs: average **90.8 words** (min 69, max 110).
5. **Temperature Comparison ("It was a dark and rainy night..."):**
   - **$T = 0.3$:** `and I had to get out of bed. I was so scared. I was so scared. I was scared. I was scared. I was scared. I was scared. I`
   - **$T = 1.2$:** `we were told to leave the apartment one hour later. It took the whole village around 8 o'clock to walk past the door. As`
   - *Observation:* At $T=0.3$, low-temperature greedy sampling traps DistilGPT2 in a repetitive degenerate loop of high-frequency n-grams. At $T=1.2$, high entropy enables rich vocabulary diversity and dynamic story progression.

---

### Task 3: Consistency Between Local PyTorch and TorchServe Handler

The generation keyword arguments, seed handling, and token slicing logic were analyzed in detail:

#### Side-by-Side Comparison:

```python
# =========================================================================
# local_inference.py (lines 111-150)
# =========================================================================
seed = clean_params.get("seed")
if seed is not None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

clean_prompt = clean_params["prompt"]
max_new_tokens = clean_params["max_new_tokens"]
temperature = clean_params["temperature"]
top_k = clean_params["top_k"]
top_p = clean_params["top_p"]

encoded = tokenizer(clean_prompt, return_tensors="pt")
input_ids = encoded["input_ids"].to(device)
attention_mask = encoded.get("attention_mask")
if attention_mask is not None:
    attention_mask = attention_mask.to(device)

input_length = input_ids.shape[1]

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

new_tokens = output_ids[0][input_length:]
generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
```

```python
# =========================================================================
# torchserve/handler.py (lines 153-192)
# =========================================================================
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

encoded = self.tokenizer(prompt, return_tensors="pt")
input_ids = encoded["input_ids"].to(self.device)
attention_mask = encoded.get("attention_mask")
if attention_mask is not None:
    attention_mask = attention_mask.to(self.device)

input_length = input_ids.shape[1]

with torch.no_grad():
    output_ids = self.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=self.tokenizer.eos_token_id,
    )

new_tokens = output_ids[0][input_length:]
generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
```

*Conclusion:* The two implementations are **100% identical in structure, arguments, and semantics**. Both enforce `min_new_tokens=1`, exact device transfers, identical random seed initialization across CPU and CUDA, and prompt token slicing (`output_ids[0][input_length:]`). In `compare_backends.py`, both backends generated identical text character-for-character.

---

### Task 4: Security and Hygiene
1. **Secret Pattern Scanning:**
   - Scanned all tracked project files for patterns: `AKIA`, `aws_secret`, `aws_access_key`, `password`, `token=`, `Bearer`.
   - **Result:** **0 matches found.** No API keys, credentials, or secrets exist in the repository.
2. **Git Ignore Configuration:**
   - Created `.gitignore` in both `story-continuation/` and project root ignoring `.venv/`, `__pycache__/`, `model_store/`, `logs/`, `*.mar`, `results/*` (preserving `.gitkeep`), `.env`, `*.pem`.
3. **Environment Variable Compliance:**
   - Verified `app.py` and `compare_backends.py` read AWS credentials via default boto3 credential provider chain (`os.environ.get("SAGEMAKER_ENDPOINT_NAME")` and `os.environ.get("AWS_REGION")`). No credentials are ever printed or logged.

---

### Task 5: Edge-Case and Resilience Testing
Created dedicated test suite [**`tests/test_edge_cases.py`**](file:///c:/Users/ratha/Desktop/NLP_Project/story-continuation/tests/test_edge_cases.py):
1. Empty prompt -> HTTP 200 with `{"error": "prompt is required"}`.
2. Missing prompt -> HTTP 200 with `{"error": "prompt is required"}`.
3. Non-string prompt -> HTTP 200 with `{"error": "prompt must be a string"}`.
4. Whitespace prompt -> HTTP 200 with `{"error": "prompt is required"}`.
5. Prompt >500 chars -> HTTP 200 with `{"error": "prompt too long (max 500 characters)"}`.
6. Out-of-range temperature ($T=2.5$ and $T=0.05$) -> HTTP 200 with boundary error.
7. Out-of-range max tokens ($500$ and $5$) -> HTTP 200 with boundary error.
8. **Worker Survival Test:** Sent invalid request followed immediately by a valid story generation request. The worker survived and generated a valid continuation.
9. **5-Request Latency Benchmark:**
   - Warm-up latency: **1,561.45 ms**
   - Subsequent latencies: `[1861.44, 1975.18, 1546.76, 1628.76, 1683.98]` ms
   - Min / Avg / Max latency: **1,546.76 ms / 1,739.22 ms / 1,975.18 ms**

---

### Task 6: Report Tables Deliverable
Generated [**`results/report_tables.md`**](file:///c:/Users/ratha/Desktop/NLP_Project/story-continuation/results/report_tables.md) containing 6 ready-to-paste tables:
1. Test cases with qualitative observations.
2. Length comparison (40 vs 120 vs 150 tokens) with word counts.
3. Temperature comparison (0.3 vs 0.8 vs 1.2 with identical seed).
4. Latency benchmarks (warm-up, min, avg, max, and factors).
5. Cross-backend comparison table from `comparison.csv`.
6. Empirical model weaknesses (repetition loops, abrupt endings, thematic drift, dialogue attribution, CPU latency scaling).

---

### Task 7: UI Demonstration Script
Generated [**`DEMO_SCRIPT.md`**](file:///c:/Users/ratha/Desktop/NLP_Project/story-continuation/DEMO_SCRIPT.md) providing exact step-by-step instructions for:
- Screenshot 1: Initial state (default layout, controls, and badge status).
- Screenshot 2: Short generation (40 tokens) with botanical garden prompt.
- Screenshot 3: Long generation (150 tokens) with captain island prompt.
- Screenshot 4: Temperature pair (0.3 vs 1.2) demonstrating repetition vs creative diversity.
- Screenshot 5: Empty prompt error handling warning banner.
- Single commands to start and stop the application.

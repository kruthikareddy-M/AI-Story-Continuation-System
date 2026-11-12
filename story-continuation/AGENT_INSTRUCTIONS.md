# MASTER INSTRUCTIONS: Story Continuation System (DistilGPT2 + PyTorch + TorchServe, SageMaker later)

You are an autonomous coding agent. Read this entire file first, then execute it from top to bottom
without waiting for me to prompt you between steps. Work in a loop: implement, run, verify, fix,
re-verify. Only stop when a stage's acceptance checks all pass, or when a stop condition in
section 3 applies.

Project root: the existing `story-continuation/` folder (Windows, PowerShell, Python 3.10 venv at `.venv`).
Everything in this file is the spec. If something here conflicts with files already in the project,
this file wins.

---

## 0. WHAT WE ARE BUILDING

A small NLP app for a university mini-project (Project 3: AI Story Continuation System).
A user types a story opening; the app generates a continuation with DistilGPT2 (PyTorch), served by
TorchServe, with a Streamlit UI. Amazon SageMaker deployment comes later (Part B, gated).

Required behaviour (from the assignment):
- Text area for the story opening.
- Continuation length option via `max_new_tokens` (Short 40, Medium 80, Long 150).
- Temperature selectable within a predefined range (0.1 to 1.5, default 0.8).
- Generation with DistilGPT2 and PyTorch (explicit tokenizer + `model.generate()`).
- Generation exposed through a custom TorchServe handler.
- Original prompt and generated continuation displayed separately.
- Same tokenizer, preprocessing and validation logic for local testing and (later) cloud inference.
- Model version, generation settings, endpoint configuration and test results must be recorded.

API contract (implement exactly):
- Request JSON: `{"prompt": "...", "max_new_tokens": 60, "temperature": 0.8, "top_k": 50, "top_p": 0.95, "seed": 42}`
  Only `prompt` is required. Others are optional and fall back to defaults.
- Success response JSON: `{"prompt": "...", "generated_text": "...", "settings": {...}, "model": "distilgpt2"}`
- Error response JSON: `{"error": "prompt is required"}` (message varies by error).

---

## 1. HARD RULES (DO and DO NOT)

DO:
- Use `AutoTokenizer`, `AutoModelForCausalLM`, `model.generate()` and `tokenizer.decode()` explicitly.
  Add short comments explaining each step (the student must explain this in a viva).
- Set `tokenizer.pad_token = tokenizer.eos_token` (GPT-2 has no pad token).
- Return ONLY the newly generated text as `generated_text` (slice off the prompt tokens before decoding).
- Put all shared defaults, limits and validation in `generation_config.py` and import it from
  `local_inference.py`, `torchserve/handler.py`, `app.py`, tests and scripts. No duplicated validation.
- Keep the pinned versions already in `requirements.txt` (torch 2.4.1 CPU, torchserve 0.12.0,
  torch-model-archiver 0.12.0, transformers 4.44.2, streamlit 1.38.0).
- Save raw command output of every verification run into `evidence/` as text files
  (for the report). Never put secrets in these files.
- Keep `PROGRESS.md` updated after every stage (see section 3).
- Write clean, commented, beginner-readable code.

DO NOT:
- Do NOT use Hugging Face `pipeline()` anywhere.
- Do NOT train or fine-tune anything. DistilGPT2 is used pretrained.
- Do NOT change pinned versions unless a stage cannot pass otherwise; if you must, record the reason in
  `PROGRESS.md` and `README.md`.
- Do NOT invent package versions or APIs. If unsure, check with `pip index versions <pkg>` / `pip show`.
- Do NOT hardcode, print, log or commit AWS keys, tokens or account IDs. Ever.
- Do NOT create or call any AWS resource in Part A. Part B starts only when the user says so.
- Do NOT delete files outside the project folder. Do NOT run destructive commands (`rm -rf`,
  `Remove-Item -Recurse` on anything outside `model_store/` build outputs or `__pycache__`).
- Do NOT leave TorchServe running when you finish a stage. Always stop it (`torchserve --stop`).
- Do NOT mark a stage done because the code "looks right". A stage is done only when its checks ran and passed.
- Do NOT skip or weaken a failing test to make it pass. Fix the code.

---

## 2. TARGET FOLDER STRUCTURE

```
story-continuation/
  AGENT_INSTRUCTIONS.md      (this file)
  PROGRESS.md                (agent state + log; create it)
  BLOCKERS.md                (only if blocked)
  README.md
  requirements.txt
  generation_config.py       (defaults, limits, validate_request)
  local_inference.py         (explicit PyTorch inference + CLI)
  save_model.py              (download + save_pretrained)
  build_mar.py               (builds model_store/story_model.mar)
  run_record.py              (writes run_record.json)
  compare_backends.py        (local PyTorch vs TorchServe, later SageMaker)
  orchestrate.py             (runs the whole local pipeline end to end)
  app.py                     (Streamlit UI)
  torchserve/
    handler.py
    config.properties
  model_store/
    distilgpt2_files/        (saved model + tokenizer)
    story_model.mar
  tests/
    __init__.py
    test_validation.py       (unit tests, no server needed)
    test_handler_unit.py     (handler preprocess/postprocess without a server)
    test_api.py              (integration tests against running TorchServe)
  results/
    test_results.csv
    comparison.csv
  evidence/                  (saved command outputs for the report)
  logs/
  sagemaker/                 (Part B only, empty until told)
```

Reconcile with what already exists: audit the project first, keep what is good, refactor what is not,
add what is missing.

---

## 3. HOW YOU OPERATE (THE LOOP)

### 3.1 State file
Maintain `PROGRESS.md` with a table of stages: `NOT STARTED | IN PROGRESS | DONE | BLOCKED`, the last
verification output summary, and a dated log of what you changed. At the start of every run, read
`PROGRESS.md` and resume from the first stage that is not DONE. Re-verify the previous DONE stage's
quick check before continuing (regressions happen).

### 3.2 Loop per stage
```
for each stage in order:
    set IN PROGRESS
    implement the stage
    run the stage's verification commands, save output to evidence/
    if all acceptance checks pass:
        set DONE, log it, continue to next stage
    else:
        read the error, find the root cause, fix it, rerun verification
        (max 5 fix attempts per stage)
        if still failing after 5: write BLOCKERS.md (error, what you tried, your best hypothesis,
        what you need from the user), set BLOCKED, STOP.
```

### 3.3 Stop conditions (stop and ask the user, do not guess)
- Java is missing or `java -version` is below 11 (tell the user to install JDK 17 and reopen the terminal).
- TorchServe cannot run on native Windows after the fallback in Stage 6 (see WSL2 note there).
- Disk space or network problems prevent downloading DistilGPT2 or packages.
- Any step would need AWS credentials, a paid resource, or a decision outside this file.
- A fix would require changing the API contract or pinned versions in a way not covered here.

### 3.4 Terminal behaviour
- Windows PowerShell. Activate the venv first: `.\.venv\Scripts\Activate.ps1`.
- Run long-lived processes (TorchServe, Streamlit) in the background, wait for readiness by polling,
  and always shut them down after checks.
- Use timeouts. If a command hangs for more than 3 minutes with no output, kill it, inspect logs in
  `logs/`, and retry once.

---

## 4. PART A: LOCAL BUILD (DO ALL OF THIS WITHOUT BEING PROMPTED)

### Stage 0: Audit and environment
Tasks:
1. List existing files. Compare to section 2. Note differences in `PROGRESS.md`.
2. Activate the venv and print versions.
3. Create missing folders and placeholder files.
4. Confirm Java: `java -version`.

Verification (save to `evidence/00_environment.txt`):
```
python --version
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
pip show torchserve torch-model-archiver | findstr /i "Name Version"
java -version
```
Accept when: Python 3.10.x, torch `2.4.1+cpu`, transformers `4.44.2`, CUDA False, torchserve and
archiver `0.12.0`, Java 11 or newer. If Java is missing, apply stop condition.

### Stage 1: Shared config and validation (`generation_config.py`)
Implement:
- `MODEL_NAME = "distilgpt2"`, `MODEL_VERSION = "1.0"`.
- `DEFAULTS = {"max_new_tokens": 60, "temperature": 0.8, "top_k": 50, "top_p": 0.95, "seed": None}`.
- `LIMITS`: `max_new_tokens` 10..200, `temperature` 0.1..1.5, `top_k` 1..100, `top_p` 0.1..1.0,
  `MAX_PROMPT_CHARS = 500`.
- `LENGTH_PRESETS = {"Short": 40, "Medium": 80, "Long": 150}`.
- `validate_request(data) -> (clean_params: dict | None, error: str | None)`:
  - not a dict -> `"request body must be a JSON object"`
  - missing, None or empty/whitespace prompt -> `"prompt is required"`
  - prompt not a string -> `"prompt must be a string"`
  - prompt longer than limit -> `"prompt too long (max 500 characters)"`
  - non-numeric or out-of-range numbers -> a clear message naming the field and allowed range
  - booleans must not be accepted as numbers
  - strip the prompt; fill defaults for missing optional fields; cast types.
Tests: `tests/test_validation.py` covering every branch above plus valid cases at the boundaries.

Accept when: `python -m unittest tests.test_validation -v` passes. Save output to
`evidence/01_validation_tests.txt`.

### Stage 2: Local PyTorch inference (`local_inference.py`)
Implement:
- `load_model()` loads `AutoTokenizer` and `AutoModelForCausalLM` for `distilgpt2`, sets pad token,
  `model.eval()`, picks device (cuda if available, else cpu). Cache so it loads once.
- `generate_story(prompt, **settings) -> dict` that:
  1. validates with `validate_request`,
  2. tokenizes (`return_tensors="pt"`, with attention mask),
  3. applies `torch.manual_seed(seed)` if a seed is given,
  4. calls `model.generate(..., do_sample=True, max_new_tokens, temperature, top_k, top_p,
     pad_token_id=tokenizer.eos_token_id, attention_mask=...)` inside `torch.no_grad()`,
  5. slices off the prompt tokens, decodes with `skip_special_tokens=True`,
  6. returns `{"prompt", "generated_text", "settings", "model"}` or `{"error": ...}`.
- CLI via argparse: `--prompt --max-new-tokens --temperature --top-k --top-p --seed`, plus `--demo`
  that runs 3 sample prompts at temperature 0.3 and 1.2 and prints results.
- Comments on every step.

Verification (save to `evidence/02_local_pytorch.txt`):
```
python local_inference.py --demo
python local_inference.py --prompt "One day, a student discovered a mysterious box in the college library..." --max-new-tokens 60 --seed 42
python local_inference.py --prompt "One day, a student discovered a mysterious box in the college library..." --max-new-tokens 60 --seed 42
python local_inference.py --prompt "" 
```
Accept when: demo prints 6 outputs; the two seeded runs produce identical text; empty prompt prints the
`prompt is required` error without a traceback; `generated_text` does not start with the prompt.

### Stage 3: Save the model (`save_model.py`)
Implement: download `distilgpt2` model and tokenizer and `save_pretrained()` into
`model_store/distilgpt2_files/`. Print the file list and total size. Skip the download if files exist
unless `--force`.

Accept when: that folder contains `config.json`, `generation_config.json`, `vocab.json`, `merges.txt`,
`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` and a weights file
(`model.safetensors` or `pytorch_model.bin`). Save listing to `evidence/03_saved_model.txt`.

### Stage 4: TorchServe handler (`torchserve/handler.py`)
Implement a class extending `ts.torch_handler.base_handler.BaseHandler` with the four stages:
- `initialize(self, context)`: `model_dir = context.system_properties.get("model_dir")`; insert
  `model_dir` into `sys.path` BEFORE importing `generation_config` (TorchServe extracts the .mar into
  that directory); load tokenizer and model from `model_dir`; set pad token; `eval()`; choose device;
  set `self.initialized = True`. Log the model name/version.
- `preprocess(self, data)`: for each request in the batch, read `row.get("body") or row.get("data")`;
  handle `bytes`/`bytearray` (decode utf-8), `str` (json.loads), and `dict`; on JSON error produce an
  error entry `"invalid JSON body"`; validate with `validate_request`; return a list of
  `{"params": ...}` or `{"error": ...}`.
- `inference(self, inputs)`: for each valid input run tokenize -> `generate` -> slice -> decode exactly as
  in `local_inference.py` (same defaults, same seed handling); pass errors through untouched; never
  raise out of the handler (wrap per-item in try/except and return `{"error": "internal error: ..."}`).
- `postprocess(self, outputs)`: return a list (one JSON-serializable dict per request) in the
  contract format.
Comment every stage clearly.

Also write `tests/test_handler_unit.py`: instantiate the handler without a server by mocking
`context`, or test `preprocess`/`postprocess` in isolation, covering bytes body, str body, dict body,
bad JSON, empty prompt and a valid request.

Accept when: `python -m unittest tests.test_handler_unit -v` passes. Save to
`evidence/04_handler_unit_tests.txt`.

### Stage 5: Package the model (`build_mar.py`)
Implement a script that:
1. ensures `model_store/distilgpt2_files/` exists (run `save_model.py` if not),
2. runs `torch-model-archiver` with: `--model-name story_model --version 1.0
   --serialized-file <weights file> --handler torchserve/handler.py
   --extra-files <config.json,generation_config.json,vocab.json,merges.txt,tokenizer.json,
   tokenizer_config.json,special_tokens_map.json,generation_config.py>
   --export-path model_store --force`
   (use full relative paths in `--extra-files`, comma separated, no spaces after commas),
3. lists the zip contents of the resulting `story_model.mar` and verifies every extra file is inside.

Accept when: `model_store/story_model.mar` exists and the listing shows all files including
`generation_config.py` and the handler. Save to `evidence/05_mar_contents.txt`.

### Stage 6: Run TorchServe locally
Write `torchserve/config.properties`:
```
inference_address=http://127.0.0.1:8080
management_address=http://127.0.0.1:8081
metrics_address=http://127.0.0.1:8082
number_of_gpu=0
default_workers_per_model=1
default_response_timeout=120
load_models=story_model.mar
model_store=model_store
disable_token_authorization=true
enable_envvars_config=true
```
(Check that every key is valid for torchserve 0.12.0; remove any that are not.)

Start command:
```
torchserve --start --ncs --model-store model_store --models story_model=story_model.mar --ts-config torchserve/config.properties
```
Poll `GET http://127.0.0.1:8080/ping` until `{"status":"Healthy"}` (timeout 120 s).

Verification (save to `evidence/06_torchserve_curl.txt`), using `curl.exe` or `Invoke-RestMethod` on Windows:
1. `GET /ping`
2. `GET http://127.0.0.1:8081/models/story_model` (shows workers)
3. POST `/predictions/story_model` with `{"prompt": "One day, a student discovered a mysterious box in the college library...", "max_new_tokens": 60, "temperature": 0.8, "seed": 42}`
4. POST with `{"prompt": ""}` -> must return the error JSON, workers must stay alive
5. POST with `{"max_new_tokens": 50}` (missing prompt)
Then `torchserve --stop`.

Accept when: ping healthy, step 3 returns a story in contract format, steps 4 and 5 return error JSON
and a following valid request still works.

Windows fallback: TorchServe support on native Windows is limited. If it fails to start or the worker
crashes for reasons you cannot fix in 5 attempts (check `logs/ts_log.log` and `logs/model_log.log`
first), do NOT keep thrashing. Write `BLOCKERS.md` with the exact error and propose WSL2 (Ubuntu) as
the fix: install JDK 17, clone/copy the project into the WSL filesystem, recreate a Python 3.10 venv,
reinstall the same pinned requirements, rerun Stage 5 and 6 there. Then stop and ask the user.

### Stage 7: Integration tests (`tests/test_api.py`)
Use `requests`; base URL from env var `TS_URL` (default `http://127.0.0.1:8080/predictions/story_model`).
Mark cleanly so the file can be run only when the server is up. Implement:
1. Assignment cases:
   - "One day, a student discovered a mysterious box in the college library..." with a short length (40)
   - "A robot entered the classroom for the first time..." with a long length (150)
   - "It was a dark and rainy night..." at temperature 0.3 and at 1.2 (same seed for both)
2. 10 different story prompts in total; each run with at least two `max_new_tokens` values (40 and 120),
   so length differences are visible.
3. Error tests: empty prompt, missing prompt, non-string prompt, whitespace-only prompt,
   prompt over 500 chars, out-of-range temperature, out-of-range max_new_tokens.
4. Record per request: timestamp, prompt (truncated for display), settings, generated text, word count,
   latency in ms, HTTP status, error. Write to `results/test_results.csv`.
5. Print a readable summary: counts passed/failed, average/min/max latency, a short table.
6. Latency note: first request after startup is slower (warm-up); report it separately from the rest.

Accept when: server-up run finishes with all expected-success cases returning `generated_text` and all
error cases returning the expected error; CSV has at least 30 rows. Save console output to
`evidence/07_api_tests.txt`.

### Stage 8: Streamlit UI (`app.py`)
Refactor the existing `app.py` (do not start from scratch) to use `generation_config.py`:
- Title, short description.
- Text area for the story opening (placeholder example).
- Length selectbox from `LENGTH_PRESETS`, temperature slider (0.1..1.5, default 0.8), optional seed box.
- Generate button with spinner; Reset button that clears inputs and outputs.
- Two separate boxes: "Original prompt" and "Generated continuation". Also show settings and latency.
- Friendly handling of empty input, validation errors, connection errors and timeouts.
- Notice: "AI-generated text from DistilGPT2. It may be repetitive, illogical or inappropriate."
- Sidebar "Backend" switch: `Local TorchServe` (HTTP POST to `TS_URL`) and `Amazon SageMaker`
  (`boto3.client("sagemaker-runtime").invoke_endpoint`, endpoint name from env var
  `SAGEMAKER_ENDPOINT_NAME`, region from `AWS_REGION`, credentials only from the standard AWS
  credential chain; if env vars are missing, show a friendly message and do nothing). The SageMaker
  path must exist and be coded, but is untested until Part B.
- No credentials in code.

Verification: start TorchServe, run `streamlit run app.py --server.headless true --server.port 8501`
in the background, confirm `http://localhost:8501/_stcore/health` returns `ok`, write a small
`tests/test_app_smoke.py` using `streamlit.testing.v1.AppTest` (available in streamlit 1.38) that
loads the app and checks the main widgets exist. Stop both servers. Save to `evidence/08_ui_smoke.txt`.
Tell the user to take the manual UI screenshots themselves.

### Stage 9: Run record and backend comparison
`run_record.py` writes `run_record.json` with: model name and version, torch/transformers/torchserve/
archiver/streamlit versions, Python version, tokenizer class, default settings, limits, `.mar` name,
version and SHA-256, TorchServe ports/workers/timeout, device, date/time, and an empty
`sagemaker` section (`endpoint_name`, `instance_type`, `container_image`, `framework_version`,
`region` as nulls).

`compare_backends.py` sends the same prompt/settings/seed to (a) local PyTorch (`local_inference`),
(b) local TorchServe, (c) SageMaker (skip with a clear message unless `SAGEMAKER_ENDPOINT_NAME` is set).
Output a table to console and `results/comparison.csv`: backend, status, response keys/format match,
settings echoed, latency, first 80 characters of text. Include a comment/printed note: with sampling,
exact wording may differ between backends; what is compared is execution, response format and
configuration. Also print whether the seeded local PyTorch and TorchServe outputs are identical (they
should be on CPU with the same seed, but do not fail if they are not; just report it).

Accept when: `run_record.json` and `results/comparison.csv` exist with correct data. Save console
output to `evidence/09_comparison.txt`.

### Stage 10: Orchestration script (`orchestrate.py`) and README
`orchestrate.py` runs the whole local pipeline with one command and writes a pass/fail summary:
```
python orchestrate.py            (everything)
python orchestrate.py --skip-download
```
Steps in order: environment check -> unit tests -> save model -> build mar -> start TorchServe ->
wait for health -> integration tests -> comparison -> run_record -> stop TorchServe (in a `finally`
block so it always stops). Print a final checklist table (PASS/FAIL per step) and exit non-zero on
any failure.

Update `README.md`: prerequisites (Python 3.10, JDK 17), setup, one-command run, manual run steps,
API contract with curl examples, folder explanation, known limitations of DistilGPT2, troubleshooting
(Java missing, port in use, handler import error, version mismatch, WSL2 note).

Final acceptance for Part A: run `python orchestrate.py` from a clean terminal; every step PASS;
`PROGRESS.md` shows all stages DONE; no TorchServe process left running
(`Get-Process java -ErrorAction SilentlyContinue` shows none from this project).

When Part A is complete, print a short summary: what passed, where the evidence files are, and the
exact list of manual items the user must still do (UI screenshots, TorchServe terminal screenshot,
report writing). Then STOP. Do not begin Part B.

---

## 5. PART B: SAGEMAKER (DO NOT START UNTIL THE USER EXPLICITLY SAYS "START PART B")

Prerequisites the user must confirm first: a faculty-approved AWS account or credits, an IAM user/role
with SageMaker + S3 permissions, credentials configured outside the project (AWS CLI profile or
environment variables), the region, and permission to create a billable endpoint.

Before writing any code, check the current AWS documentation for: the PyTorch inference container
versions available in SageMaker, how the SageMaker PyTorch container runs TorchServe via the
inference toolkit (`model_fn`, `input_fn`, `predict_fn`, `output_fn` in `code/inference.py`, or a
custom handler), and instance pricing. Pick a PyTorch version supported by the container that is
as close as possible to 2.4.1 and record the exact versions chosen and why. Do not guess.

Planned stages:
- B1: build `model.tar.gz` (model files + `code/inference.py` that reuses `generation_config.py` and the
  same tokenizer/generate logic; JSON in, JSON out in the same contract).
- B2: upload to S3 (bucket name from user), create `sagemaker/deploy.py` (SageMaker Python SDK
  `PyTorchModel`, small CPU instance such as `ml.m5.large`, endpoint name `story-continuation-endpoint`).
  Make the TorchServe usage explicit and documented.
- B3: `sagemaker/invoke_test.py` using `boto3` `invoke_endpoint`; reuse `tests/test_api.py` cases against
  the endpoint; record latency (note cold start vs warm).
- B4: wire `app.py` SageMaker backend, update `compare_backends.py` and `run_record.json`.
- B5: `sagemaker/cleanup.py` deleting the endpoint, endpoint config, model, and (if user agrees) the S3
  objects, with a confirmation prompt and a verification listing saved to `evidence/`.
- ALWAYS ask the user for a go-ahead right before creating the endpoint, and remind them to run
  cleanup right after the demo. Never leave the endpoint running when you finish.

---

## 6. FINAL OUTPUT CHECKLIST (what the student submits)

- [ ] Source code, `requirements.txt`, README with local run instructions
- [ ] `torchserve/handler.py`, `config.properties`, `story_model.mar` build details
- [ ] Evidence of local PyTorch inference (`evidence/02_*`)
- [ ] Evidence of TorchServe API testing (`evidence/06_*`, `evidence/07_*`)
- [ ] `results/test_results.csv` (10+ prompts, different lengths and temperatures, error cases, latency)
- [ ] `run_record.json` and `results/comparison.csv`
- [ ] UI screenshots (manual)
- [ ] Part B: SageMaker deployment/invocation evidence with account IDs hidden, and cleanup proof
- [ ] Report and presentation (manual; the agent may help draft after the code is done)

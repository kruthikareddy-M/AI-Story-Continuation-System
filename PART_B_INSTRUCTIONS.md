# PART B: DEPLOY TO AMAZON SAGEMAKER (TorchServe model server)

Read AGENT_INSTRUCTIONS.md first. Its HARD RULES (section 1) and LOOP (section 3) still apply here.
Part A is finished. This file replaces the Part B outline in AGENT_INSTRUCTIONS.md.

Goal: host the same DistilGPT2 story generator on a SageMaker real-time endpoint that runs TorchServe
inside the AWS PyTorch inference container, invoke it from Python and from the Streamlit UI, record
results, then DELETE everything.

---

## 0. PART 1: THINGS THE STUDENT MUST DO MANUALLY (the agent cannot and must not)

Complete these before telling the agent to start. Tick each one.

- [ ] Faculty-approved AWS account or credits confirmed. Know the approved REGION.
- [ ] Billing safety: in AWS Billing, create a budget alert (for example 5 USD) and turn on email alerts.
- [ ] Create an IAM user (or use the one faculty gave) with permissions for SageMaker, S3 and
      CloudWatch Logs. Create an access key for CLI use. Never paste keys into chat, code or screenshots.
- [ ] Install the AWS CLI and run `aws configure` (or `aws configure --profile story`). Verify with
      `aws sts get-caller-identity` (do NOT screenshot this: it shows the account ID).
- [ ] Create a SageMaker EXECUTION ROLE in IAM: trusted entity = SageMaker, permissions =
      AmazonSageMakerFullAccess plus read/write access to the S3 bucket below. Copy its ARN.
- [ ] Create one private S3 bucket in the same region (example name: `story-continuation-<yourname>-<random>`).
- [ ] Set these environment variables in the PowerShell session the agent will use (values are yours; never
      write them into any project file):
      ```
      $env:AWS_REGION = "<approved-region>"
      $env:AWS_PROFILE = "<profile-if-you-use-one>"
      $env:SAGEMAKER_ROLE_ARN = "<execution-role-arn>"
      $env:S3_BUCKET = "<bucket-name>"
      $env:SAGEMAKER_ENDPOINT_NAME = "story-continuation-endpoint"
      ```
- [ ] If a later step fails with ResourceLimitExceeded for the instance type, open Service Quotas and
      request a quota increase for "ml.m5.large for endpoint usage" (or ask faculty).

Then tell the agent: "START PART B". The agent works through stages B0 to B6. It must STOP and wait for
the words "APPROVE DEPLOY" before B3 (this is when billing starts), and "APPROVE CLEANUP" is NOT needed:
cleanup in B6 is mandatory and runs right after testing, with a confirmation prompt.

---

## 1. WHAT THE SAGEMAKER SETUP LOOKS LIKE

- The AWS PyTorch inference container has TorchServe preinstalled, and the SageMaker PyTorch inference
  toolkit starts TorchServe inside it.
- In SageMaker, our `torchserve/handler.py` is not used directly. The toolkit wraps four functions from an
  `inference.py` file: `model_fn(model_dir)`, `input_fn(request_body, content_type)`,
  `predict_fn(input_object, model)`, `output_fn(prediction, accept)`. They map to our handler stages:
  `initialize` -> `model_fn`, `preprocess` -> `input_fn`, `inference` -> `predict_fn`,
  `postprocess` -> `output_fn`.
- The assignment requires the TorchServe model server to be explicit. Show it: document this mapping in
  the README and capture the TorchServe startup lines from the endpoint's CloudWatch logs (B3).
- Reuse the SAME generation logic and validation (`generation_core.py` if it exists, otherwise create it
  now by extracting the shared code from `local_inference.py`, and keep `generation_config.py`). No second
  copy of the logic.

Chosen container (verify, do not assume): PyTorch 2.4 inference CPU image for SageMaker, Python 3.11.
Local is torch 2.4.1 / torchserve 0.12.0 / Python 3.10, so the closest match is framework_version "2.4",
py_version "py311". AWS notes say that image line was updated to TorchServe 0.12.0, but confirm against
the container's own logs in B3 and record what you actually see.

---

## 2. HARD RULES FOR PART B

DO:
- Pin the SageMaker Python SDK to v2: `pip install "sagemaker<3"` and record the exact version. (The
  documented `PyTorchModel` flow used below is v2. Do not migrate to v3 unless a check fails.)
- Read credentials ONLY from the standard AWS chain (profile/env vars). Read region, role ARN, bucket
  and endpoint name ONLY from the environment variables listed above.
- Mask account IDs in everything you save: replace the 12-digit account number with `<ACCOUNT_ID>` in
  evidence files, logs, and run_record.json. Never save the role ARN in full: mask the account part.
- Use ONE instance, ONE endpoint, instance type `ml.m5.large`, name from `SAGEMAKER_ENDPOINT_NAME`.
- Keep the endpoint up only as long as needed: finish all invocation tests and screenshots help in one
  sitting, then clean up.
- Save outputs into `evidence/B*_*.txt`.

DO NOT:
- Do NOT create any resource before "APPROVE DEPLOY".
- Do NOT create more than one endpoint, use GPU instances, enable autoscaling, or create anything outside
  the project's named resources.
- Do NOT print, store or commit credentials, session tokens, or unmasked account IDs.
- Do NOT leave any endpoint, endpoint config or model behind when you stop, even if a stage failed.
  If anything fails after resources were created, run cleanup first, then report.
- Do NOT edit local TorchServe behaviour from Part A except to share code through `generation_core.py`;
  rerun `python orchestrate.py --skip-download` after any such change.

---

## 3. STAGES

### B0: Preflight (no AWS cost)
1. Confirm env vars exist (print only whether they are set, never their values).
2. `pip install "sagemaker<3" boto3`; print versions; add to requirements.txt as a separate
   `requirements-sagemaker.txt` (do not disturb the pinned Part A requirements).
3. Check credentials: run `aws sts get-caller-identity` through boto3, print only the masked ARN type
   (for example `arn:aws:iam::<ACCOUNT_ID>:user/...`). If it fails, STOP and tell the student.
4. Verify the container exists for the region:
   ```python
   import sagemaker
   uri = sagemaker.image_uris.retrieve(framework="pytorch", region=REGION, version="2.4",
                                       py_version="py311", image_scope="inference",
                                       instance_type="ml.m5.large")
   ```
   Print the URI with the account ID masked. If this raises "unsupported", list the supported versions
   that the SDK reports (look at `sagemaker.image_uris` config or error text), choose the nearest
   PyTorch version (2.x, CPU, inference), record the change and the reason in PROGRESS.md and continue.
5. Check S3 bucket access (head_bucket) and that the bucket region equals AWS_REGION.
6. Check which instance quota is likely needed: print a reminder, do not call paid APIs.
Accept: all checks pass. Save to `evidence/B0_preflight.txt`.

### B1: Build the inference code (no AWS cost)
Create `sagemaker/code/` containing:
- `inference.py` with:
  - `model_fn(model_dir)`: load AutoTokenizer and AutoModelForCausalLM from `model_dir`, set pad token,
    `eval()`, device cpu/cuda, return a small object or dict holding model, tokenizer, device.
  - `input_fn(request_body, content_type)`: accept `application/json` only; decode bytes; `json.loads`;
    on bad JSON or wrong content type return a dict `{"_error": "..."}` (do not raise, so the contract
    stays the same as local).
  - `predict_fn(input_object, model_bundle)`: if `_error` present return `{"error": ...}`; else
    `validate_request`, then call the SAME shared generation function used locally; return the
    contract dict `{"prompt","generated_text","settings","model"}` or `{"error": ...}`.
  - `output_fn(prediction, accept)`: return `json.dumps(prediction)` for `application/json`.
  - Comments mapping each function to its TorchServe-handler stage.
- `generation_config.py` and `generation_core.py` (copied by the build script, never hand-edited here).
- `requirements.txt` containing only what the container lacks, pinned: `transformers==4.44.2`
  (verify it installs on Python 3.11; if not, choose the nearest compatible version and record it).
  Do NOT list torch (the container provides it).
Create `sagemaker/build_model_tar.py` that:
- copies the saved model files from `model_store/distilgpt2_files/` into a temp folder at the ROOT of
  the archive (config.json, tokenizer files, weights),
- creates `sagemaker/model.tar.gz` (model files only; code is attached by the SDK via `source_dir`),
- prints the file list, size, and SHA-256.
Create `tests/test_sagemaker_inference_local.py` that imports `sagemaker/code/inference.py` and calls
`model_fn` (pointing at `model_store/distilgpt2_files`), `input_fn`, `predict_fn`, `output_fn` directly
for: a valid request with a seed, an empty prompt, missing prompt, invalid JSON, wrong content type.
Also assert that with the same seed the output text equals the local `local_inference.py` output.
Accept: the test passes, tar built. Save to `evidence/B1_local_inference_py_test.txt`.

### B2: Upload to S3 (tiny cost)
Upload `sagemaker/model.tar.gz` to `s3://$S3_BUCKET/story-continuation/model.tar.gz`. Print the S3 URI
(bucket name is fine to show). Verify with `head_object` (size matches).
Save to `evidence/B2_s3_upload.txt`.

### B3: Deploy the endpoint (BILLING STARTS) — WAIT FOR "APPROVE DEPLOY"
Before asking, print: region, instance type, endpoint name, container URI (masked), and the reminder
that billing runs per hour while the endpoint is InService, plus that cleanup is mandatory.
After approval, write `sagemaker/deploy.py`:
```python
from sagemaker.pytorch import PyTorchModel
model = PyTorchModel(
    model_data=S3_URI, role=ROLE_ARN, entry_point="inference.py", source_dir="sagemaker/code",
    framework_version="2.4", py_version="py311",   # use the version chosen in B0
    sagemaker_session=session,
)
predictor = model.deploy(initial_instance_count=1, instance_type="ml.m5.large",
                         endpoint_name=ENDPOINT_NAME)
```
- Add environment variables if needed for timeouts (for example
  `SAGEMAKER_MODEL_SERVER_TIMEOUT` set to "120"); only keep ones that you confirm are valid by checking
  the container logs or documentation.
- Wait until the endpoint is InService (poll `describe_endpoint`, max 15 minutes). If it goes Failed,
  read the CloudWatch log group for the endpoint, print the error, run cleanup, then fix and ask for
  approval again (maximum 2 deploy attempts in total; after that STOP and report).
- Pull the endpoint's CloudWatch log lines that prove TorchServe is the model server (search for
  "TorchServe", "Torch worker", "Model server started"). Save the matching lines to
  `evidence/B3_torchserve_in_cloudwatch.txt` (mask account IDs) and record the TorchServe version seen.
- Save `describe_endpoint`, `describe_endpoint_config` (instance type, count, variant) to
  `evidence/B3_endpoint_config.txt`, masked.
Accept: status InService, TorchServe lines captured.

### B4: Invoke and test the endpoint
Write `sagemaker/invoke_test.py` using `boto3.client("sagemaker-runtime").invoke_endpoint`
(ContentType and Accept `application/json`). Run, against the live endpoint:
- the same 31 test cases as `tests/test_api.py` (reuse the case definitions, do not duplicate them):
  3 assignment cases, 10 prompts at two lengths, and all error cases;
- the 5-request latency run (first request reported separately as cold/warm-up).
Save `results/sagemaker_test_results.csv` (same columns as local plus the backend name) and the console
summary to `evidence/B4_sagemaker_tests.txt`.
Accept: every expected-success case returns a contract-format story; every error case returns the
expected error JSON; no unexpected exceptions.

### B5: Integrate and compare
- Finish the SageMaker backend in `app.py` (endpoint name and region from env vars only). Smoke test it
  with a script or AppTest by selecting the SageMaker backend and sending one prompt. The student will
  take the manual UI screenshots.
- Update `compare_backends.py` so it now runs all three backends (local PyTorch, local TorchServe,
  SageMaker) with the same prompt, settings and seed. Print and save `results/comparison.csv`. Report whether
  seeded outputs are identical across backends, and if not, state that the compared items are
  execution, response format, and configuration. Do not fail the stage on text differences.
- Update `run_record.py` so `run_record.json` fills the `sagemaker` section: endpoint name, instance
  type, instance count, container image (masked), framework version, py version, TorchServe version
  seen in logs, SDK version, region, model artifact S3 URI (bucket name only) and SHA-256.
- Create `results/report_tables_part_b.md`: latency table (local TorchServe vs SageMaker, with
  warm-up separate), three-backend comparison table, and a short list of differences observed.
Save console outputs to `evidence/B5_*.txt`.

### B6: MANDATORY CLEANUP (run immediately after B4/B5, even if earlier stages failed)
Write and run `sagemaker/cleanup.py`:
1. Ask for confirmation by printing exactly what will be deleted (endpoint, endpoint config, model)
   and requiring the student to type `DELETE`.
2. Delete the endpoint, its endpoint configuration, and the model.
3. Ask separately (type `DELETE S3`) before deleting the uploaded `model.tar.gz` from S3. The bucket
   itself is NOT deleted by the script.
4. Verify: `list_endpoints`, `list_endpoint_configs`, `list_models` filtered by the project name return
   nothing. Print the results.
5. Save to `evidence/B6_cleanup.txt` (masked).
Then print the reminder for the student to check the AWS console (SageMaker > Inference > Endpoints) and
take a screenshot showing no endpoints, and to check Billing next day.

---

## 4. KNOWN PROBLEMS AND FIXES

- ResourceLimitExceeded: quota for the instance type is zero. Request a quota increase.
- Endpoint Failed or stuck Creating: read CloudWatch logs for `/aws/sagemaker/Endpoints/<name>`.
  Common causes: `transformers` fails to install from `requirements.txt` (check pin), model files
  missing from the tar root, handler import error (shared modules not in `source_dir`).
- 413 or timeout errors: keep `max_new_tokens` at 200 or less; invocation requests must finish well under
  the real-time endpoint limit (about 60 seconds). Set the model server timeout env var if needed.
- AccessDenied: the execution role cannot read the S3 bucket, or the IAM user lacks SageMaker permissions.
- ValidationException about role: the ARN is wrong or the role is not trusted by SageMaker.
- Different outputs from local: expected with sampling. Compare format, settings and execution, and use the
  seed only as a best effort across different hardware.

## 5. FINAL ACCEPTANCE FOR PART B

- Endpoint was deployed, invoked successfully from script and UI, and CloudWatch evidence shows TorchServe.
- `results/sagemaker_test_results.csv`, `results/comparison.csv`, `run_record.json` are complete.
- Cleanup verified (nothing left in SageMaker), `evidence/B6_cleanup.txt` saved.
- PROGRESS.md updated; no secrets or unmasked account IDs anywhere in the project (run a final search
  for 12-digit numbers, AKIA, aws_secret and report the result).
Then STOP and print what the student still must do manually: AWS console screenshots (endpoint
InService with the account ID hidden, endpoint configuration, and the empty endpoint list after
cleanup), UI screenshots using the SageMaker backend, and the report.

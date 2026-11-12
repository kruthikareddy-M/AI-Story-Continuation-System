# AI Story Continuation System - UI Demonstration & Screenshot Guide

This guide provides a step-by-step walkthrough for manually demonstrating the web application and capturing required UI screenshots for the Project 3 report.

---

## ⚡ 1. Starting & Stopping the Application

### How to Start
Open a terminal in the project directory and run:

**Option A: Windows Batch File (Double-click or terminal)**
```cmd
run.bat
```

**Option B: PowerShell**
```powershell
.\run.ps1
```

**Option C: Direct Python**
```powershell
python run_app.py
```

*The script will automatically start TorchServe on port 8085, wait for health confirmation, and open the Streamlit UI in your browser at `http://localhost:8501`.*

### How to Stop
When finished taking screenshots, press **`Ctrl + C`** in the terminal where `run.bat` / `run.ps1` is running.
TorchServe will be cleanly stopped automatically.

To verify or force stop manually at any time:
```powershell
.\.venv\Scripts\python.exe -m ts.model_server --stop
```

---

## 📸 2. Step-by-Step Screenshot Checklist

### Screenshot 1: Initial State (Default Web UI)
*Goal: Demonstrate clean UI layout, backend selector, input text area, and hyperparameter controls.*

1. **Open Browser:** Navigate to `http://localhost:8501`.
2. **Settings in Sidebar:**
   - **Inference Backend:** Select `Local TorchServe` *(green badge should show `Online & Healthy`)*.
   - **Continuation Length:** Leave at default `Medium (80 tokens)`.
   - **Sampling Temperature:** Leave at default slider value `0.80`.
   - **Top-k / Top-p:** Ensure `50` and `0.95`.
3. **Main Text Area:**
   - Leave default sample opening in the text area:
     ```
     One day, a student discovered a mysterious box in the college library...
     ```
4. **Capture Screenshot:** Capture the full browser window showing the title, sidebar controls, and empty results container before clicking Generate.
   - *Save as:* `screenshot_01_initial_state.png`

---

### Screenshot 2: Short Generation (40 Tokens)
*Goal: Demonstrate fast, concise story continuation with the Short preset.*

1. **Sidebar Settings:**
   - **Continuation Length:** Select `Short (40 tokens)`.
   - **Sampling Temperature:** Keep at `0.80`.
   - **Random Seed:** Type `42` (ensures reproducible output).
2. **Story Opening Prompt:**
   Enter the following prompt into the text area:
   ```
   In an overgrown botanical garden, a strange mechanical flower began to
   ```
3. **Action:** Click the large primary **"✨ Generate Story Continuation"** button.
4. **Wait:** ~1-2 seconds for generation to complete.
5. **Verify Display:**
   - Yellow callout box displays: `PROMPT: In an overgrown botanical garden...`
   - Green callout box displays the newly generated continuation.
   - Metrics card displays latency and token count.
6. **Capture Screenshot:** Capture the browser window showing both prompt and continuation callouts.
   - *Save as:* `screenshot_02_short_generation.png`

---

### Screenshot 3: Long Generation (150 Tokens)
*Goal: Demonstrate extended narrative generation using the Long preset.*

1. **Sidebar Settings:**
   - **Continuation Length:** Select `Long (150 tokens)`.
   - **Sampling Temperature:** Keep at `0.80`.
   - **Random Seed:** Type `42`.
2. **Story Opening Prompt:**
   Enter the following prompt into the text area:
   ```
   The captain looked through the telescope and realized the island was
   ```
3. **Action:** Click **"✨ Generate Story Continuation"**.
4. **Wait:** ~6-8 seconds for 150 tokens to generate on CPU.
5. **Verify Display:**
   - Continuation spans multiple sentences with full narrative development.
   - Word count should read over 100 words.
6. **Capture Screenshot:** Capture the browser window showing the extended generated story.
   - *Save as:* `screenshot_03_long_generation.png`

---

### Screenshot 4: Temperature Comparison (Low vs High Pair)
*Goal: Illustrate the impact of temperature on creativity vs repetitive degeneracy.*

#### Part A: Low Temperature (0.30 - Deterministic / Repetitive)
1. **Sidebar Settings:**
   - **Continuation Length:** Select `Medium (80 tokens)`.
   - **Sampling Temperature Slider:** Slide to `0.30`.
   - **Random Seed:** Type `99`.
2. **Prompt:**
   ```
   It was a dark and rainy night...
   ```
3. **Action:** Click **"✨ Generate Story Continuation"**.
4. **Capture Screenshot:** Notice the repetitive loops (*"I was so scared. I was so scared..."*).
   - *Save as:* `screenshot_04a_temp_0.3.png`

#### Part B: High Temperature (1.20 - Creative / Diverse)
1. **Sidebar Settings:**
   - Keep the same prompt: `It was a dark and rainy night...`
   - Keep the same seed: `99`.
   - **Sampling Temperature Slider:** Slide up to `1.20`.
2. **Action:** Click **"✨ Generate Story Continuation"**.
3. **Capture Screenshot:** Notice the rich, varied vocabulary and unexpected story direction (*"we were told to leave the apartment one hour later..."*).
   - *Save as:* `screenshot_04b_temp_1.2.png`

---

### Screenshot 5: Error Handling (Empty Prompt Validation)
*Goal: Prove client and server schema validation handles invalid input gracefully without crashing.*

1. **Main Text Area:**
   - Clear the prompt completely (delete all characters so the box is completely empty).
2. **Action:** Click **"✨ Generate Story Continuation"**.
3. **Verify Display:**
   - A warning notification appears: `⚠️ Prompt is required. Please type a story opening.`
   - No exception or backend crash occurs.
4. **Capture Screenshot:** Capture the screen with the red/amber validation message clearly visible.
   - *Save as:* `screenshot_05_error_validation.png`

---

## 📋 Summary of Screenshot Deliverables for Your Report

| Screenshot File | Section / Purpose | Expected Content |
| :--- | :--- | :--- |
| `screenshot_01_initial_state.png` | System Overview | Full UI with sidebar, backend status, and initial inputs. |
| `screenshot_02_short_generation.png` | Length Controls | Short (40-token) continuation with prompt/output separation. |
| `screenshot_03_long_generation.png` | Scalability & Pacing | Long (150-token) multi-sentence story continuation. |
| `screenshot_04a_temp_0.3.png` | Qualitative Analysis | Low temperature ($T=0.3$) showing greedy loop behavior. |
| `screenshot_04b_temp_1.2.png` | Qualitative Analysis | High temperature ($T=1.2$) showing diverse creative output. |
| `screenshot_05_error_validation.png` | Error Handling & Safety | Empty prompt rejection with clean warning banner. |

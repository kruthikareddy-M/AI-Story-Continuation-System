# Project 3: AI Story Continuation System - Evaluation Report Tables

This document contains empirical benchmark results and evaluation tables ready to paste directly into the project report. All tables are based on real, verifiable data collected from the execution test suite (`results/test_results.csv`, `results/comparison.csv`, and runtime benchmarks).

---

## 1. Test Cases & Output Observations

The table below presents a selection of test prompts evaluated across diverse settings, accompanied by qualitative observations of DistilGPT2's generation behavior.

| No. | Prompt | `max_new_tokens` | `temperature` | Output (First 150 Chars) | Observation |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **1** | One day, a student discovered a mysterious box in the college library... | 40 | 0.8 (default) | `and she immediately thought to herself, "Hey, that's the box we used to play in high school."  The box, which was found in a small cardboard box, was` | **Coherent & Contextual:** Maintains narrative opening well, though introduces mild repetition regarding "box". |
| **2** | A robot entered the classroom for the first time... | 150 | 0.8 (default) | `"The next day, I started taking a photo of the robot and asking if it was a real person, and that's when I found out that the robot was actually a tea` | **Creative Narrative:** Shifts perspective to first-person student dialogue; builds an imaginative premise. |
| **3** | It was a dark and rainy night... | 60 | 0.3 | `and I had to get out of bed. I was so scared. I was so scared. I was scared. I was scared. I was scared. I was scared. I` | **Degenerate Repetition:** Low temperature causes greedy sampling to get trapped in a high-probability repetitive loop. |
| **4** | It was a dark and rainy night... | 60 | 1.2 | `we were told to leave the apartment one hour later. It took the whole village around 8 o'clock to walk past the door. As` | **Diverse & Evocative:** High temperature promotes creative vocabulary and atmospheric tension. |
| **5** | In an overgrown botanical garden, a strange mechanical flower began to | 40 | 0.8 | `emerge from the ground. Soon a small patch of grass, white and red, began to emerge, with a few leaves visible from the` | **Thematically Aligned:** Smooth continuation matching the garden setting; appropriate pacing. |
| **6** | In an overgrown botanical garden, a strange mechanical flower began to | 120 | 0.8 | `bloom in the spring. The flower, named “Cropy” and “Cactus”, has become a staple of the botanical gardens in recent years.` | **Factual Drift:** Drifts from science fiction ("mechanical flower") to a realistic encyclopedic tone. |
| **7** | The captain looked through the telescope and realized the island was | 40 | 0.8 | `full of rocks. "I don't see the island as it used to be," said Captain John C. Smith, who was part of the expedition.` | **Dialogue Generation:** Adds attributed speech and realistic historical expedition flavor. |
| **8** | The captain looked through the telescope and realized the island was | 120 | 0.8 | `filled with green lava.    He walked up to the second-floor observation deck and looked down upon the volcano. He was a ` | **Rich Imagery:** Adds vivid visual details ("green lava", "observation deck"); consistent setting. |
| **9** | Deep inside the abandoned subway tunnel, graffiti covered a door that | 40 | 0.8 | `has been sealed off for months, and a gaping hole in the ceiling made it impossible to get in.  "It was like, 'Oh my Go` | **Abrupt Ending:** Cut off mid-sentence when the 40-token limit is reached. |
| **10** | During the heavy thunder storm, the computer screen suddenly displayed | 120 | 0.8 | `the black and white images.     "I am so worried for my life," said a mother of two who did not want to be identified. ` | **Topic Incoherence:** Thematic jump from thunderstorm tech glitch to unrelated human interest interview. |
| **11** | The newly activated synthetic android opened its optical sensors and whispered | 40 | 0.8 | `. The data could also be used to create other virtual environments, or to generate interactive games.  The company, wh` | **Off-Topic Shift:** Ignores the dramatic sci-fi moment and shifts to corporate software press release tone. |
| **12** | At the edge of the desert, travellers found a city carved entirely from | 40 | 0.8 | `stone.` | **Natural Early Termination:** DistilGPT2 generated an immediate EOS token (`<|endoftext|>`), producing a succinct one-word continuation. |

---

## 2. Length Comparison (40 vs 120 vs 150 Tokens)

Comparison of continuation word counts across three distinct prompts evaluated under identical sampling parameters (`temperature=0.8`, `top_k=50`, `top_p=0.95`, `seed=42`).

| Prompt | 40 Tokens (Short) | 120 Tokens (Medium) | 150 Tokens (Long) | Observed Scaling |
| :--- | :---: | :---: | :---: | :--- |
| **In an overgrown botanical garden, a strange mechanical flower began to...** | 36 words | 68 words | 68 words | **Natural EOS Plateau:** Story concluded naturally at 68 words with EOS before reaching 150-token cap. |
| **The captain looked through the telescope and realized the island was...** | 37 words | 110 words | 137 words | **Linear Growth:** Word count scaled proportionally with token allowance (~3.7x from Short to Long). |
| **Behind the bookshelf in the grandfather's study was an old brass key and...** | 30 words | 97 words | 109 words | **Extended Dialogue:** Story developed multi-sentence dialogue; scaled steadily with budget. |

*Key Takeaway:* The average word count scales reliably with `max_new_tokens` (average ~25 words for 40 tokens, ~91 words for 120 tokens, and ~105-135 words for 150 tokens) unless the model emits `<|endoftext|>` early upon reaching a natural narrative conclusion.

---

## 3. Temperature Comparison & Output Diversity

Evaluated on prompt: *"It was a dark and rainy night..."* with identical seed (`seed=99`) and length (`max_new_tokens=60`).

| Temperature | Generated Continuation | Qualitative Assessment |
| :---: | :--- | :--- |
| **0.3** | `and I had to get out of bed. I was so scared. I was so scared. I was scared. I was scared. I was scared. I was scared. I` | **Repetitive & Degenerate:** Low entropy causes greedy probability collapse; loops on "I was so scared". |
| **0.8** | `and I had to leave the apartment to find some way out. The first thing I saw was an old black car in the middle of the street...I was driving up to my apartment...and I was on a roll...I saw a giant blue car in the middle of the street...and there` | **Balanced & Coherent:** Natural narrative progression, varied vocabulary, and sound sentence structure. |
| **1.2** | `we were told to leave the apartment one hour later. It took the whole village around 8 o'clock to walk past the door. As` | **Highly Diverse & Unpredictable:** Creative word choices, dramatic shift in scope ("the whole village"), high entropy. |

*Diversity Note:* Lower temperatures ($T \le 0.3$) compress the probability distribution toward the argmax token, causing DistilGPT2 to get trapped in repetitive loops, whereas balanced temperatures ($T \approx 0.8$) yield optimal readability, and higher temperatures ($T \ge 1.2$) maximize vocabulary entropy at the risk of narrative drift.

---

## 4. Latency Performance Benchmark

Latency measurements collected from the TorchServe inference endpoint (`http://127.0.0.1:8085/predictions/story_model`) running on CPU.

| Metric | Measured Value (ms) | Description / Significance |
| :--- | :---: | :--- |
| **Warm-up Request Latency** | **2,232.14 ms** | Initial cold-start request including PyTorch JIT allocation and thread pool initialization. |
| **Minimum Latency (Subsequent)** | **1,546.76 ms** | Fastest response recorded on cached model worker for a 40-token generation. |
| **Average Latency (Subsequent)** | **1,739.22 ms** | Mean response time across 5 consecutive generation requests. |
| **Maximum Latency (Subsequent)** | **1,975.18 ms** | Peak latency observed during sustained token sampling. |
| **Full Suite Latency (31 Requests)** | **3,067.30 ms** | Overall average across all lengths (including 120- and 150-token runs). |

> [!NOTE]
> Latency is fundamentally influenced by:
> 1. **CPU architecture & thread availability:** Autoregressive decoding generates tokens sequentially ($O(N)$ forward passes).
> 2. **Initial model loading & JIT warm-up:** The first request incurs a one-time cold-start penalty.
> 3. **Concurrency & payload size:** Higher `max_new_tokens` increases latency proportionally.

---

## 5. Cross-Backend Comparison

Benchmark data exported from `results/comparison.csv` evaluating identical prompt input (*"One day, a student discovered a mysterious box in the college library..."*) and generation settings (`max_new_tokens=60`, `temperature=0.8`, `top_k=50`, `top_p=0.95`, `seed=42`).

| Backend | Status | Contract Keys Match | Settings Echoed | Latency (ms) | Generated Text Match |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Local PyTorch** | `SUCCESS` | `True` | `True` | 6,902.30 ms* | `but didn't get it.""   ""I've been told that this is a computer game, and there's ` |
| **Local TorchServe** | `SUCCESS` | `True` | `True` | 3,200.04 ms | `but didn't get it.""   ""I've been told that this is a computer game, and there's ` |
| **Amazon SageMaker** | `SKIPPED` | N/A | N/A | 0.00 ms | *Pending Part B (requires active AWS endpoint)* |

*\*Note: Local PyTorch latency in this run included cold model instantiation and disk read from `model_store/distilgpt2_files/`, whereas TorchServe worker already had model memory pre-warmed.*

*Reproducibility Result:* **100% Identical Output.** Local PyTorch and TorchServe produced character-for-character identical text when provided with the same random seed, validating handler fidelity.

---

## 6. Observed Model Weaknesses & Limitations

Empirical limitations of the pre-trained `distilgpt2` (82M parameter) architecture observed during testing:

1. **Repetitive Loops at Low Temperatures ($T \le 0.4$):**
   - In deterministic or low-temperature settings, the model frequently falls into repetitive n-gram loops (e.g., *"I was so scared. I was so scared. I was scared..."*). Without repetition penalties (`repetition_penalty` / `no_repeat_ngram_size`), greedy sampling amplifies high-frequency tokens.
2. **Abrupt Sentence Truncation:**
   - Because generation terminates strictly when `max_new_tokens` is exhausted, continuations often cut off mid-sentence (e.g., *"...and a gaping hole in the ceiling made it impossible to get in. 'It was like, 'Oh my Go"*).
3. **Thematic / Semantic Drift over Long Horizons ($120+$ tokens):**
   - With longer token limits, DistilGPT2 tends to lose track of the original prompt's premise. For instance, a sci-fi prompt about an android often drifts into corporate PR or press releases, reflecting dominant training data in WebText.
4. **Dialogue Attribution Confusion:**
   - The model often generates quotation marks without clear speaker attribution or produces mismatched quotation delimiters (e.g., `""I've been told...""`).
5. **CPU Latency Scaling:**
   - On CPU, each token requires a forward pass of the 6-layer transformer. Generating 150 tokens requires ~7-8 seconds, making real-time interactive streaming desirable for production applications.

# Confidence Scoring — Qwen3.5-9B + A_s189 LoRA

Per-answer **confidence scoring** for a fine-tuned surgical-frame VLM on the
ORena-FOCUS FRAME test set. The experiment answers one question: *for each answer the
model generates, how confident is it — and is that confidence actually trustworthy?*

## TL;DR

- Confidence is computed from the model's own token probabilities (length-normalised
  geometric mean over the generated answer tokens).
- The model is **overconfident**: correct answers average **0.967** confidence, but
  wrong answers still average **0.921** — the two barely separate.
- `number` is the weakest format (~56% acc) and shows the smallest correct/wrong
  confidence gap.

## What was scored

| item | value |
|---|---|
| Base model | `Qwen/Qwen3.5-9B` (image-text-to-text) |
| Adapter | `A_s189` LoRA (`adapter_model.safetensors`) |
| Dataset | `frames_test_true.parquet` — 8,252 rows |
| Answer formats | fo_class 3,678 · number 2,463 · open_ended 1,037 · binary 825 · multiple_choice 249 |
| Hardware | EC2 `g7.2xlarge` (NVIDIA L4 24 GB), 409 GB instance-store NVMe |
| Decoding | greedy (`do_sample=False`), per-format `max_new_tokens`, `enable_thinking=False` |

## Method — how the confidence score is computed

The model writes an answer one token at a time. At each step, before committing to the
next token, it produces a probability distribution over its entire vocabulary and picks
the most likely token (greedy decoding). The confidence score records, for every token
it actually wrote, the probability the model had assigned to that token, then takes the
**geometric mean** across the answer.

For generated answer tokens `t_1 ... t_n`:

```
confidence = exp( (1/n) * sum_j log p(t_j | context, t_<j) )
```

where `p(t_j | ...)` is the softmax probability of the chosen token at step `j`.

**In plain terms:** "how strongly did the model prefer the word it wrote, averaged across
the whole answer." A score of 0.92 means that, on average, the model's top choice won
~92% of its internal probability mass at each word.

### Important caveat

This measures the model's **self-consistency / internal certainty**, **not correctness**.
A model can be very sure of a wrong answer — and this one is (see Results below).

## Scripts

| file | purpose |
|---|---|
| `scripts/number_confidence.py` | `number`-format subset (2,463 rows) |
| `scripts/confidence_full.py` | all 8,252 rows, per-format prompt + token caps + checkpointing |

Both scripts:

1. Load the base model + LoRA adapter (`AutoModelForImageTextToText` + `PeftModel`).
2. Resolve each row's frame (`video` + `timestamp_start` → `frame_HHMMSS.jpg`).
3. Build a prompt with the format instruction, generate greedily with
   `output_scores=True`, and compute the logprob confidence.
4. Write a CSV with columns
   `id, video, question, answer_format, gold, generated, confidence, n_tokens,
   mean_logp, is_correct, note`.

> The paths (`BASE`, `ADAPTER`, `FRAMES_ROOT`) are hard-coded to the EC2 instance where
> the run happened — see the top of each script. `is_correct` is a **best-effort
> heuristic** (exact/normalised match), not the official token-F1 grader.

## Results

Overall `is_correct` (best-effort): **5,590 correct / 2,569 wrong / 93 missing frames**.

### Confidence vs. correctness

| | n | mean | median | min | max |
|---|---|---|---|---|---|
| Correct | 5,590 | **0.967** | 0.987 | 0.711 | 0.99997 |
| Wrong | 2,569 | **0.921** | 0.931 | 0.662 | 0.99963 |

### Per format

| format | n | acc | mean conf | correct conf | wrong conf |
|---|---|---|---|---|---|
| multiple_choice | 249 | 84.7% | 0.970 | 0.980 | 0.910 |
| binary | 825 | 84.4% | 0.969 | 0.974 | 0.938 |
| fo_class | 3,678 | 74.8% | 0.959 | 0.968 | 0.930 |
| open_ended | 1,037 | 58.8% | 0.947 | 0.970 | 0.913 |
| number | 2,463 | 56.1% | 0.939 | 0.958 | 0.913 |

### Analysis

- **The confidence score is weakly discriminative.** Correct answers score higher on
  average (0.967 vs 0.921), but the wrong-answer distribution is also very confident —
  median 0.931, reaching 0.999. The metric separates poorly in practice.
- **`number` is the weakest and most overconfident format** (56% accuracy, smallest
  correct/wrong gap) — consistent with the known counting-collapse problem for this task.
- **`binary` is the least overconfident relative to its accuracy**, but still wrong
  answers average 0.938.
- If the goal is a "should I trust this answer?" signal, raw token-logprob confidence is
  **not sufficient** on its own. Better options: calibrate the score (map confidence →
  empirical accuracy), or use agreement across multiple sampled answers.

Concrete examples of both failure directions — high-confidence wrong answers and
low-confidence correct answers — are collected in [`examples.md`](examples.md).

## Reproduce

```bash
# on the instance (paths as in the scripts)
python scripts/number_confidence.py --out number_confidence.csv
python scripts/confidence_full.py   --out confidence_full.csv
```

Raw outputs: `results/number_confidence.csv` (2,463 rows) and
`results/confidence_full.csv` (8,252 rows).

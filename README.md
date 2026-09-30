# RL base — GRPO / RLVR for surgical-frame VQA (SageMaker BYOC)

Base version of the reinforcement-learning scripts for the ORena-FOCUS FRAME track.
This branch (`rl-byoc-sagemaker`) is an **orphan branch**: it has a single root commit
and shares no history with `main` (which holds the confidence-scoring experiment). It
contains only the RL scaffold.

**Goal:** further-improve a LoRA-tuned surgical VLM with RLVR + GRPO, packaged as a
SageMaker **BYOC** (Bring Your Own Container) training job.

## Stack

| choice | value |
|---|---|
| RL algorithm | GRPO (Group Relative Policy Optimization) via RLVR |
| Framework | **TRL** (`GRPOTrainer`) + **vLLM** (optional fast rollouts) |
| Base model | `Qwen/Qwen3.5-9B` (`AutoModelForImageTextToText`) |
| Starting adapter | `A_s189` LoRA (SFT checkpoint) |
| Reward | verifiable, per-format, deterministic (see below) |
| Packaging | Docker image → ECR → SageMaker training job |

## Repo layout

```
reward.py             # verifiable reward (per-format scorer) as a TRL callable
data.py               # parquet + frames  ->  TRL GRPO dataset (prompt + image)
train_grpo.py         # main GRPO training script (loads base + adapter, runs GRPO)
fo_defs.txt           # foreign-object class definitions (REPLACE with your FO_definitions.txt)
Dockerfile            # BYOC training image (PyTorch + TRL + transformers)
requirements.txt      # core deps (fast, reliable base)
requirements-vllm.txt # adds vLLM (only needed for --use-vllm)
train                 # container entrypoint (maps SageMaker SM_HP_* -> argparse)
run_local.sh          # convenience: run on a local / EC2 GPU box
build_and_push.sh     # build image + push to ECR
sagemaker_launch.py   # submit the training job via boto3
```

## The reward (verifiable / RLVR)

`reward.py:reward_func` ports the per-format scoring from the swift reward plugin to a
plain TRL callable. It is **deterministic** — the whole point of RLVR is a reward you
can trust without a learned reward model.

| format | reward |
|---|---|
| `number` | exact match of first integer (`0.0` / `1.0`) |
| `binary` | exact match on yes/no (`0.0` / `1.0`) |
| `multiple_choice` | exact match (`0.0` / `1.0`) |
| `fo_class` | **micro-F1** over the 8 canonical classes |
| `open_ended` | **token-F1** (bag-of-words F1) |

The `<answer></answer>` **format gate is removed** (the merged model answers without the
tags, and the hard gate zeroed every reward). `extract_answer` still honors the tags
when present, and falls back to the raw completion otherwise. Re-introduce a gate only
after a cold-start SFT warmup that makes the model reliably emit tags.

Canonical `fo_class` labels (fixed — do not drop `Gallstone`):

`Clip`, `Silicone loop`, `Needle`, `Sponge`, `Specimen bag`, `Specimen`,
`External drain`, `Gallstone`

## Quickstart (local / single GPU)

```bash
python train_grpo.py \
  --base /path/to/Qwen3.5-9B \
  --adapter /path/to/A_s189 \
  --parquet /path/to/frames_train_true.parquet \
  --frames-root /path/to/curated_frames \
  --fo-defs fo_defs.txt \
  --output-dir ./outputs/grpo \
  --max-steps 200
```

### Two model-strategy paths

**Path A — merge vision, GRPO the language layers (recommended, CoA-aligned).**
Perception isn't the bottleneck; mapping observations → terminology is. Merge the SFT
adapter into the base, then RL-train a fresh LoRA on language attention only.

```bash
python train_grpo.py --base ... --adapter .../A_s189 \
  --merge-adapter --lora
```

**Path B — plain GRPO from the base (heaviest, most vLLM-friendly).**

```bash
python train_grpo.py --base ... --parquet ... --frames-root ... \
  --use-vllm            # full weights, no live LoRA
```

### vLLM caveat (important)

vLLM fast rollouts do **not** support a live LoRA on vision layers. Concretely:
- `--use-vllm` works cleanly only when the policy has **no active vision LoRA**
  (Path A after `--merge-adapter`, or Path B full weights).
- If you keep a live vision LoRA, use transformers rollouts (omit `--use-vllm`).

`requirements-vllm.txt` is separate so the base image builds fast; install it only when
enabling `--use-vllm`.

## SageMaker BYOC

### 1. Build + push

```bash
./build_and_push.sh <aws_account_id> us-east-1
```

### 2. Stage data on S3

Put the parquet and the frames tree under one S3 prefix, e.g.
`s3://your-bucket/rl/train/` (the training job mounts it at
`/opt/ml/input/data/train/`). Point `--parquet` and `--frames-root` at those paths.

### 3. Launch

```bash
python sagemaker_launch.py \
  --image-uri <ecr>.dkr.ecr.us-east-1.amazonaws.com/rl-byoc-grpo:latest \
  --role <SageMakerExecutionRoleArn> \
  --s3-train s3://your-bucket/rl/train/ \
  --s3-output s3://your-bucket/rl/output/
```

SageMaker hyperparameters are passed as env vars (`SM_HP_*`) and mapped to the
script's flags by the `train` entrypoint. Booleans (`--merge-adapter`, `--lora`,
`--use-vllm`) take `true`/`false`.

## Caveats / TODOs

- **LoRA `target_modules`** default to language-layer attention projections
  (`q_proj, k_proj, v_proj, o_proj`). Extend (e.g. add MLP, `visual.*`, `merger`) only
  if you deliberately want to RL-tune perception too.
- **TRL VLM column convention**: this scaffold uses an `image` column plus `prompt`
  messages with `{"type": "image"}`. If your TRL version uses `images` (list), adjust
  `data.py` — it is the only place the convention appears.
- **`max_prompt_length`**: frames inflate the prompt length. Tune it (default 4096)
  against your image resolution.
- **Reward shaping**: rewards are currently coarse (exact match / F1). A length- or
  format-penalty term is easy to add in `reward.py:score` once a real run is stable.
- Replace `fo_defs.txt` with the project's actual `FO_definitions.txt`.

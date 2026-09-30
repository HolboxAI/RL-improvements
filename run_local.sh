#!/usr/bin/env bash
set -e
# Local / EC2 single-GPU run. Override the paths, then append extra flags if needed.
python train_grpo.py \
  --base /path/to/Qwen3.5-9B \
  --adapter /path/to/A_s189 \
  --parquet /path/to/frames_train_true.parquet \
  --frames-root /path/to/curated_frames \
  --fo-defs fo_defs.txt \
  --output-dir ./outputs/grpo \
  --merge-adapter --lora \
  "$@"

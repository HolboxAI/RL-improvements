#!/usr/bin/env python
"""GRPO / RLVR training for surgical-frame VQA (TRL + optional vLLM).

Loads the base VLM (Qwen3.5-9B) and optionally the SFT LoRA adapter (A_s189), then
runs GRPO with a deterministic per-format reward (see reward.py).

Model-strategy flags:
    --adapter        load the SFT LoRA as the policy starting point
    --merge-adapter  merge the adapter into the base (perception locked in)
    --lora           add a fresh trainable LoRA on language attention only (default)
    --use-vllm       fast vLLM rollouts (see README: no live vision LoRA)

Recommended (CoA-aligned): --adapter A_s189 --merge-adapter --lora
Heaviest / most vLLM-friendly: no adapter, full weights, --use-vllm
"""
from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

from data import build_dataset
from reward import reward_func

# Language-layer attention projections only -> "GRPO the language layers, not vision".
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model path/ID (Qwen3.5-9B)")
    ap.add_argument("--adapter", default=None, help="SFT LoRA adapter dir (e.g. A_s189)")
    ap.add_argument("--merge-adapter", action="store_true",
                    help="merge the adapter into the base weights before RL")
    ap.add_argument("--lora", action="store_true",
                    help="add a fresh trainable LoRA for the RL step")
    ap.add_argument("--lora-targets", default=",".join(DEFAULT_LORA_TARGETS),
                    help="comma-separated LoRA target module suffixes")
    ap.add_argument("--parquet", required=True, help="train parquet path")
    ap.add_argument("--frames-root", required=True, help="frames tree root")
    ap.add_argument("--fo-defs", default="fo_defs.txt")
    ap.add_argument("--output-dir", default="./outputs/grpo")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all rows")
    # GRPO knobs
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=128)
    ap.add_argument("--max-prompt-length", type=int, default=4096)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--use-vllm", action="store_true")
    args = ap.parse_args()

    fo_defs = open(args.fo_defs, encoding="utf-8").read().strip() if os.path.exists(args.fo_defs) else ""

    print("loading processor + base model", flush=True)
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True)

    peft_config = None
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        if args.merge_adapter:
            print("merging adapter into base weights", flush=True)
            model = model.merge_and_unload()

    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
        )
        print(f"fresh RL LoRA on: {peft_config.target_modules}", flush=True)

    ds = build_dataset(args.parquet, args.frames_root, fo_defs,
                       args.max_samples or None)
    print(f"dataset: {len(ds)} examples", flush=True)

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=0.5,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=ds,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"DONE -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

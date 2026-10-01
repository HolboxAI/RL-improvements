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


def _str2bool(v: str) -> bool:
    """Parse SageMaker script-mode 'true'/'false' and bare-flag usage."""
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model path/ID (Qwen3.5-9B)")
    ap.add_argument("--adapter", default=None, help="SFT LoRA adapter dir (e.g. A_s189)")
    ap.add_argument("--device_map", default=None, help="device_map for base model load (e.g. auto)")
    ap.add_argument("--merge_adapter", nargs="?", const=True, default=False, type=_str2bool,
                    help="merge the adapter into the base weights before RL")
    ap.add_argument("--lora", nargs="?", const=True, default=False, type=_str2bool,
                    help="add a fresh trainable LoRA for the RL step")
    ap.add_argument("--lora_targets", default=",".join(DEFAULT_LORA_TARGETS),
                    help="comma-separated LoRA target module suffixes")
    ap.add_argument("--parquet", required=True, help="train parquet path")
    ap.add_argument("--frames_root", required=True, help="frames tree root")
    ap.add_argument("--fo_defs", default="fo_defs.txt")
    ap.add_argument("--output_dir", default="./outputs/grpo")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all rows")
    # GRPO knobs
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--max_completion_length", type=int, default=128)
    ap.add_argument("--max_prompt_length", type=int, default=4096)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--resume_from_checkpoint", default=None,
                    help="checkpoint dir to resume from; default: auto-detect /opt/ml/checkpoints")
    ap.add_argument("--model_dir", default=None, help="final artifact dir (e.g. /opt/ml/model)")
    ap.add_argument("--use_vllm", nargs="?", const=True, default=False, type=_str2bool)
    args = ap.parse_args()

    fo_defs = open(args.fo_defs, encoding="utf-8").read().strip() if os.path.exists(args.fo_defs) else ""

    print("loading processor + base model", flush=True)
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map=args.device_map)

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
        save_steps=args.save_steps,
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

    # Resume: explicit path, else the newest checkpoint SageMaker restored to /opt/ml/checkpoints.
    resume = args.resume_from_checkpoint
    if resume is None:
        ckpt_root = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
        if os.path.isdir(ckpt_root):
            subs = [d for d in os.listdir(ckpt_root)
                    if os.path.isdir(os.path.join(ckpt_root, d)) and d.startswith("checkpoint-")]
            if subs:
                resume = os.path.join(ckpt_root, sorted(subs, key=lambda d: int(d.split("-")[-1]))[-1])
                print(f"resuming from {resume}", flush=True)

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    if args.model_dir:
        trainer.save_model(args.model_dir)
        print(f"final artifact -> {args.model_dir}", flush=True)
    print(f"DONE -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

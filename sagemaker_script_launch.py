#!/usr/bin/env python
"""Submit the GRPO/RLVR smoke test via SageMaker script mode (no Docker/ECR).

Fast path to a first result: uses SageMaker's managed PyTorch CUDA image and
pip-installs requirements.txt at job start. Spot + checkpoint/resume enabled.

The BYOC image (build_and_push.sh) remains the later vLLM path. Hyperparameter
keys must match train_grpo.py's argparse flags exactly (underscores).
"""
from __future__ import annotations

import argparse

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

HYPERPARAMETERS = {
    "base": "Qwen/Qwen3.5-9B",               # pulled from HF at job start
    "adapter": "/opt/ml/input/data/adapter",  # A_s189 channel
    "device_map": "auto",                     # shard policy+ref across 4xA10G
    "merge_adapter": "true",
    "lora": "true",
    "parquet": "/opt/ml/input/data/data/frames_train_rl.parquet",
    "frames_root": "/opt/ml/input/data/frames",
    "fo_defs": "/opt/ml/code/fo_defs.txt",
    "output_dir": "/opt/ml/checkpoints",      # synced to S3 -> resume
    "model_dir": "/opt/ml/model",             # final artifact -> output S3
    "max_samples": "512",
    "num_generations": "4",
    "max_steps": "50",
    "save_steps": "25",
    "per_device_batch_size": "1",
    "grad_accum": "4",
    "lr": "1e-6",
    "max_completion_length": "128",
    "max_prompt_length": "4096",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, help="SageMaker execution role ARN")
    ap.add_argument("--instance-type", default="ml.g5.12xlarge")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--name", default="rl-grpo-spot-smoke")
    ap.add_argument("--source-dir", default="/tmp/RL-improvements")
    ap.add_argument("--framework-version", default="2.5.1")
    ap.add_argument("--py-version", default="py311")
    ap.add_argument("--checkpoint-s3", default="s3://stanford-train-data/rlvr/checkpoints")
    ap.add_argument("--output-path", default="s3://stanford-train-data/rlvr/smoke-output")
    ap.add_argument("--max-run", type=int, default=3600)
    ap.add_argument("--max-wait", type=int, default=7200)
    ap.add_argument("--on-demand", action="store_true")
    args = ap.parse_args()

    boto_session = boto3.Session(profile_name="stanford_gpu", region_name=args.region)
    session = sagemaker.Session(boto_session=boto_session, default_bucket="stanford-train-data")

    est = PyTorch(
        entry_point="train_grpo.py",
        source_dir=args.source_dir,
        role=args.role,
        instance_count=1,
        instance_type=args.instance_type,
        framework_version=args.framework_version,
        py_version=args.py_version,
        hyperparameters=HYPERPARAMETERS,
        output_path=args.output_path,
        base_job_name=args.name,
        sagemaker_session=session,
        max_run=args.max_run,
        max_wait=args.max_wait,
        use_spot_instances=not args.on_demand,
        checkpoint_s3_uri=args.checkpoint_s3,
        checkpoint_local_path="/opt/ml/checkpoints",
        volume_size=200,
    )

    est.fit(inputs={
        "adapter": "s3://stanford-train-data/models/A_s189",
        "frames": "s3://stanford-train-data/curated_frames",
        "data": "s3://stanford-train-data/rlvr/frames_train_rl.parquet",
    }, wait=False)

    print("JOB:", est.latest_training_job.name)


if __name__ == "__main__":
    main()

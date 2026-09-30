#!/usr/bin/env python
"""Submit the BYOC GRPO training job to SageMaker via boto3 (no SDK dependency)."""
from __future__ import annotations

import argparse

import boto3

# Hyperparameters are exposed to the container as SM_HP_* env vars; the `train`
# entrypoint maps them back to train_grpo.py flags.
HYPERPARAMETERS = {
    "base": "/opt/ml/input/data/train/models/Qwen3.5-9B",
    "adapter": "/opt/ml/input/data/train/models/A_s189",
    "parquet": "/opt/ml/input/data/train/frames_train_true.parquet",
    "frames_root": "/opt/ml/input/data/train/curated_frames",
    "fo_defs": "/opt/ml/code/fo_defs.txt",
    "output_dir": "/opt/ml/model",
    "merge_adapter": "true",
    "lora": "true",
    "max_steps": "200",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-uri", required=True)
    ap.add_argument("--role", required=True, help="SageMaker execution role ARN")
    ap.add_argument("--s3-train", required=True, help="s3://... prefix with parquet + frames + models")
    ap.add_argument("--s3-output", required=True, help="s3://... output prefix")
    ap.add_argument("--instance-type", default="ml.p4d.24xlarge")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--name", default="rl-grpo-byoc")
    args = ap.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    resp = sm.create_training_job(
        TrainingJobName=args.name,
        AlgorithmSpecification={
            "TrainingImage": args.image_uri,
            "TrainingInputMode": "File",
        },
        RoleArn=args.role,
        ResourceConfig={
            "InstanceCount": 1,
            "InstanceType": args.instance_type,
            "VolumeSizeInGB": 200,
        },
        HyperParameters=HYPERPARAMETERS,
        InputDataConfig=[{
            "ChannelName": "train",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": args.s3_train,
                "S3DataDistributionType": "FullyReplicated",
            }},
        }],
        OutputDataConfig={"S3OutputPath": args.s3_output},
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
    )
    print(resp["TrainingJobArn"])


if __name__ == "__main__":
    main()

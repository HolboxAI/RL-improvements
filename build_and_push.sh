#!/usr/bin/env bash
set -e
# Usage: ./build_and_push.sh <aws_account_id> [region]
ACCOUNT_ID="${1:?usage: build_and_push.sh <aws_account_id> [region]}"
REGION="${2:-us-east-1}"
IMAGE="rl-byoc-grpo"
ECR="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE}"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ECR"

aws ecr describe-repositories --repository-names "$IMAGE" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$IMAGE" --region "$REGION"

docker build -t "$IMAGE" .
docker tag "$IMAGE:latest" "$ECR:latest"
docker push "$ECR:latest"
echo "pushed $ECR:latest"

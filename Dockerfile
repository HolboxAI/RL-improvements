# SageMaker BYOC training image for GRPO/RLVR (TRL).
# PyTorch + CUDA come from the base; TRL/transformers/peft are pip-installed.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . /opt/ml/code
WORKDIR /opt/ml/code
RUN chmod +x train

# SageMaker BYOC runs the container with the `train` program; hyperparameters arrive
# as SM_HP_* env vars, which `train` maps to train_grpo.py flags.
ENTRYPOINT ["/opt/ml/code/train"]

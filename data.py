"""Parquet + frames -> TRL GRPO dataset for surgical-frame VQA.

Each example becomes:
    {
        "prompt": [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}],
        "image": <PIL.Image>,          # the resolved frame
        "answer": <gold string>,       # passed to reward_func as a kwarg
        "answer_format": <format>,     # passed to reward_func as a kwarg
        "video": <video name>,         # provenance only
    }

The frame path is resolved the same way as the confidence scripts:
    video "X - Y.avi"          -> dir "X - Y"
    timestamp_start "00:28:22" -> frame_002822.jpg
"""
from __future__ import annotations

import os

import pandas as pd
from datasets import Dataset
from PIL import Image

PROMPT_TMPL = (
    "You are a surgical assistant. You are given an endoscopic frame from a "
    "minimally invasive procedure. Analyze the image and answer the surgical "
    "question based on the visual evidence. Be precise and concise.\n\n"
    "{fo_defs}\n\n"
    "Question: {question}\n"
    "Expected Format: {fmt}\n"
    "Instruction: {instruction}\n\n"
    "Return ONLY the answer. Do NOT explain. Do NOT add any text before or after it."
)

FORMAT_INSTRUCTION = {
    "binary": "Return only yes or no.",
    "number": "Return only a non-negative integer.",
    "percentage": "Return only a percentage number.",
    "fo_class": "Return only the foreign object class name.",
    "open_ended": "Return a concise answer.",
    "multiple_choice": "Return only one option from: top/left, top/right, bottom/left, bottom/right.",
    "time": "Return only in hh:mm:ss format.",
    "matching": "Return only the exact matching text.",
}

FRAME_SUBDIRS = [
    "First_Test_Frames", "Second_Test_Frames",
    "First_Train_Frames", "Second_Train_Frames",
]


def resolve_frame(video: str, timestamp_start, frames_root: str):
    """Map (video, timestamp) -> frame jpg path, or None if missing."""
    folder = os.path.splitext(video)[0]
    label = str(timestamp_start).replace(" ", "").replace(":", "")
    fname = f"frame_{label}.jpg"
    for sub in FRAME_SUBDIRS:
        p = os.path.join(frames_root, sub, folder, fname)
        if os.path.exists(p):
            return p
    return None


def build_dataset(parquet_path: str, frames_root: str, fo_defs: str,
                  max_samples: int | None = None) -> Dataset:
    """Build the TRL GRPO dataset, skipping rows whose frame is missing."""
    df = pd.read_parquet(parquet_path)
    if max_samples:
        df = df.head(max_samples)

    records = []
    for _, r in df.iterrows():
        path = resolve_frame(r["video"], r["timestamp_start"], frames_root)
        if path is None:
            continue
        fmt = str(r["answer_format"])
        instruction = FORMAT_INSTRUCTION.get(fmt, "Return only the final answer.")
        prompt_text = PROMPT_TMPL.format(
            fo_defs=fo_defs, question=r["question"], fmt=fmt, instruction=instruction)
        records.append({
            "prompt": [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}],
            "image": Image.open(path).convert("RGB"),
            "answer": str(r["answer"]),
            "answer_format": fmt,
            "video": r["video"],
        })

    return Dataset.from_list(records)

#!/usr/bin/env python
"""
Confidence scoring for `number` answer-format questions with Qwen3.5-9B + A_s189 LoRA.

For each `number` row: resolve frame, generate the answer greedily, and compute
token-logprob confidence = exp(mean log-prob of the generated answer tokens)
(length-normalized, so short and long answers are comparable).

Output CSV columns: id, video, question, gold, generated, confidence, n_tokens,
mean_logp, is_correct, note
"""
import argparse, math, os, re, time
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE = "/opt/dlami/nvme/orena/models/Qwen3.5-9B"
ADAPTER = "/opt/dlami/nvme/orena/models/A_s189"
FRAMES_ROOT = "/home/ubuntu/curated_frames"
FRAMES_DIRS = [
    os.path.join(FRAMES_ROOT, "First_Test_Frames"),
    os.path.join(FRAMES_ROOT, "Second_Test_Frames"),
    os.path.join(FRAMES_ROOT, "First_Train_Frames"),
    os.path.join(FRAMES_ROOT, "Second_Train_Frames"),
]
FO_DEFS = open("/home/ubuntu/FO_definitions.txt", encoding="utf-8").read().strip()


def build_prompt(question):
    return (
        "You are a surgical assistant. You are given an endoscopic frame from a "
        "minimally invasive procedure. Analyze the image and answer the surgical "
        "question based on the visual evidence. Be precise and concise.\n\n"
        f"{FO_DEFS}\n\n"
        f"Question: {question}\n"
        "Expected Format: number\n"
        "Instruction: Return only a non-negative integer.\n\n"
        "Return ONLY the answer. Do NOT explain. Do NOT add any text before or after it."
    )


_video_root_cache = {}


def timestamp_to_frame_path(video_name, ts):
    video_folder = os.path.splitext(video_name)[0]      # strip .avi/.mp4
    label = ts.replace(" ", "").replace(":", "")
    fname = f"frame_{label}.jpg"
    root = _video_root_cache.get(video_folder)
    if root is not None:
        return os.path.join(root, video_folder, fname)
    for r in FRAMES_DIRS:
        p = os.path.join(r, video_folder, fname)
        if os.path.exists(p):
            _video_root_cache[video_folder] = r
            return p
    for r in FRAMES_DIRS:
        d = os.path.join(r, video_folder)
        if os.path.isdir(d):
            _video_root_cache[video_folder] = r
            return os.path.join(d, fname)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="limit rows (0 = all)")
    ap.add_argument("--parquet", default="/home/ubuntu/frames_test_true.parquet")
    ap.add_argument("--out", default="/opt/dlami/nvme/orena/number_confidence.csv")
    args = ap.parse_args()

    print("loading processor/model...", flush=True)
    processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    print("model ready", flush=True)

    df = pd.read_parquet(args.parquet)
    num = df[df.answer_format == "number"].reset_index(drop=True)
    if args.limit:
        num = num.head(args.limit)
    print(f"scoring {len(num)} number rows", flush=True)

    rows = []
    t0 = time.time()
    for i, r in num.iterrows():
        rec = dict(id=int(r["id"]), video=r["video"], question=r["question"],
                   gold=str(r["answer"]), generated="", confidence=None,
                   n_tokens=0, mean_logp=None, is_correct=None, note="")
        p = timestamp_to_frame_path(r["video"], r["timestamp_start"])
        if p is None or not os.path.exists(p):
            rec["note"] = "missing frame"
            rows.append(rec)
            continue
        try:
            img = Image.open(p).convert("RGB")
            prompt_text = build_prompt(r["question"])
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ]}]
            kw = dict(add_generation_prompt=True, tokenize=True,
                      return_dict=True, return_tensors="pt")
            try:
                inputs = processor.apply_chat_template(messages, enable_thinking=False, **kw)
            except TypeError:
                inputs = processor.apply_chat_template(messages, **kw)
            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     return_dict_in_generate=True, output_scores=True)
            prompt_len = inputs["input_ids"].shape[-1]
            gen_ids = out.sequences[0][prompt_len:]
            answer = processor.decode(gen_ids, skip_special_tokens=True).strip()
            logps = [torch.log_softmax(out.scores[j][0], dim=-1)[gen_ids[j]].item()
                     for j in range(len(gen_ids))]
            conf = math.exp(sum(logps) / len(logps)) if logps else 1.0
            rec["generated"] = answer
            rec["confidence"] = round(conf, 6)
            rec["n_tokens"] = len(logps)
            rec["mean_logp"] = round(sum(logps) / len(logps), 6) if logps else None
            # lightweight correctness: first integer in generated vs gold
            m = re.search(r"\d+", answer)
            if m:
                try:
                    rec["is_correct"] = int(m.group(0)) == int(r["answer"])
                except ValueError:
                    rec["is_correct"] = None
        except Exception as e:
            rec["note"] = f"error: {e}"
        rows.append(rec)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(num)} done  ({el:.0f}s elapsed, {el/(i+1):.2f}s/row)", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"DONE wrote {args.out} with {len(out_df)} rows", flush=True)


if __name__ == "__main__":
    main()

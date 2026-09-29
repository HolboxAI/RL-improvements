#!/usr/bin/env python
"""
Confidence scoring for ALL answer formats on the full test set with
Qwen3.5-9B + A_s189 LoRA.

Same token-logprob confidence as number_confidence.py, but generalized:
  - runs every row (all answer_format values)
  - per-format prompt instruction + max_new_tokens
  - per-format best-effort is_correct heuristic (NOT the official grader)

Output CSV: id, video, question, answer_format, gold, generated, confidence,
n_tokens, mean_logp, is_correct, note
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

FORMAT_TOKENS = {"binary": 8, "number": 16, "fo_class": 32,
                 "multiple_choice": 48, "open_ended": 96}


def get_format_instruction(fmt):
    return {
        "binary": "Return only yes or no.",
        "number": "Return only a non-negative integer.",
        "percentage": "Return only a percentage number.",
        "fo_class": "Return only the foreign object class name.",
        "open_ended": "Return a concise answer.",
        "multiple_choice": "Return only one option from: top/left, top/right, bottom/left, bottom/right.",
        "time": "Return only in hh:mm:ss format.",
        "matching": "Return only the exact matching text.",
    }.get(fmt, "Return only the final answer.")


def build_prompt(question, fmt):
    instruction = get_format_instruction(fmt)
    return (
        "You are a surgical assistant. You are given an endoscopic frame from a "
        "minimally invasive procedure. Analyze the image and answer the surgical "
        "question based on the visual evidence. Be precise and concise.\n\n"
        f"{FO_DEFS}\n\n"
        f"Question: {question}\n"
        f"Expected Format: {fmt}\n"
        f"Instruction: {instruction}\n\n"
        "Return ONLY the answer. Do NOT explain. Do NOT add any text before or after it."
    )


def check_correct(generated, gold, fmt):
    g = (generated or "").strip()
    if fmt == "number":
        m = re.search(r"\d+", g)
        return int(m.group(0)) == int(gold) if m else False
    if fmt == "binary":
        return g.lower().rstrip(".") == gold.strip().lower().rstrip(".")
    if fmt == "fo_class":
        def norm(s):
            return {x.strip().lower() for x in re.split(r"[,;]| and ", s) if x.strip()}
        return norm(g) == norm(gold)
    # default: normalized exact match
    def norm2(s):
        return re.sub(r"\s+", " ", s.strip().lower().strip("."))
    return norm2(g) == norm2(gold)


_video_root_cache = {}


def timestamp_to_frame_path(video_name, ts):
    video_folder = os.path.splitext(video_name)[0]
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
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--parquet", default="/home/ubuntu/frames_test_true.parquet")
    ap.add_argument("--out", default="/opt/dlami/nvme/orena/confidence_full.csv")
    args = ap.parse_args()

    print("loading processor/model...", flush=True)
    processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    print("model ready", flush=True)

    df = pd.read_parquet(args.parquet).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"scoring {len(df)} rows", flush=True)
    print(df["answer_format"].value_counts(dropna=False).to_string(), flush=True)

    rows = []
    t0 = time.time()
    for i, r in df.iterrows():
        fmt = str(r["answer_format"])
        rec = dict(id=int(r["id"]), video=r["video"], question=r["question"],
                   answer_format=fmt, gold=str(r["answer"]), generated="",
                   confidence=None, n_tokens=0, mean_logp=None, is_correct=None, note="")
        p = timestamp_to_frame_path(r["video"], r["timestamp_start"])
        if p is None or not os.path.exists(p):
            rec["note"] = "missing frame"
            rows.append(rec)
            continue
        try:
            img = Image.open(p).convert("RGB")
            prompt_text = build_prompt(r["question"], fmt)
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
            max_new = FORMAT_TOKENS.get(fmt, 64)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
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
            try:
                rec["is_correct"] = check_correct(answer, rec["gold"], fmt)
            except Exception:
                rec["is_correct"] = None
        except Exception as e:
            rec["note"] = f"error: {e}"
        rows.append(rec)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(df)} done  ({el:.0f}s elapsed, {el/(i+1):.2f}s/row)", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)  # periodic checkpoint

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"DONE wrote {args.out} with {len(out_df)} rows", flush=True)


if __name__ == "__main__":
    main()

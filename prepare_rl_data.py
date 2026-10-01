#!/usr/bin/env python
"""Part-A data prep for RLVR on the surgical-frame VQA train set.

Makes the deterministic reward trustworthy by fixing the labels, then removes rows
that can't produce a clean training signal:

  1. canonicalize `answer` to exactly what reward.py parses
     (number/binary/multiple_choice/fo_class are already clean; open_ended is not)
  2. drop rows whose canonical answer is empty
  3. drop exact content duplicates (same frame + question + answer)
  4. drop label-conflict prompt groups (same frame + question, different gold)
  5. drop rows with no resolvable frame (optional, needs a frames manifest)

Outputs:
  <out>.parquet     cleaned train set
  <out>.dropped.parquet   every dropped row + reason
  <out>.report.txt        human-readable change log
"""
from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd

CANONICAL_FO = {
    "Clip", "Silicone loop", "Needle", "Sponge",
    "Specimen bag", "Specimen", "External drain", "Gallstone",
}
MC_OPTIONS = {"top/left", "top/right", "bottom/left", "bottom/right"}
FRAME_SUBDIRS = [
    "First_Train_Frames", "Second_Train_Frames",
    "First_Test_Frames", "Second_Test_Frames",
]
ERRORS_RE = re.compile(r"\bErrors?\s*:", re.IGNORECASE)


# --- per-format canonicalizers -------------------------------------------------

def clean_number(s: str) -> str:
    m = re.search(r"-?\d+", str(s))
    return str(int(m.group(0))) if m else (s or "").strip()


def clean_binary(s: str) -> str:
    return (s or "").strip().lower().rstrip(".")


def clean_multiple_choice(s: str) -> str:
    return (s or "").strip().lower()


def clean_fo_class(s: str) -> str:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    return ", ".join(parts)


def clean_open_ended(s: str) -> str:
    s = (s or "").strip()
    m = ERRORS_RE.search(s)          # cut "Errors: <self-correction notes>" leakage
    if m:
        s = s[:m.start()]
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" .!?,;:")
    return s.lower().strip()


CLEANERS = {
    "number": clean_number,
    "binary": clean_binary,
    "multiple_choice": clean_multiple_choice,
    "fo_class": clean_fo_class,
    "open_ended": clean_open_ended,
}


def load_frames(path: str) -> set:
    """Parse an `aws s3 ls --recursive` manifest into {(subdir, folder, fname)}."""
    frames = set()
    with open(path) as f:
        for line in f:
            key = " ".join(line.split()[3:])     # key may contain spaces
            parts = key.split("/")
            if len(parts) >= 3:
                frames.add((parts[1], parts[2], parts[-1]))
    return frames


def resolve_frame(video: str, ts, frames: set):
    folder = os.path.splitext(video)[0]
    label = str(ts).replace(" ", "").replace(":", "")
    fname = f"frame_{label}.jpg"
    for sub in FRAME_SUBDIRS:
        if (sub, folder, fname) in frames:
            return f"{sub}/{folder}/{fname}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/Users/vishnunair/Downloads/frames_train_true.parquet")
    ap.add_argument("--out", default="/Users/vishnunair/Downloads/frames_train_rl.parquet")
    ap.add_argument("--frames-manifest", default=None,
                    help="optional `aws s3 ls --recursive` manifest for frame resolution")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet).reset_index(drop=True)
    n0 = len(df)
    report = [f"input rows: {n0}"]

    # 1. canonicalize
    df["answer_raw"] = df["answer"].astype(str)
    df["answer"] = df.apply(lambda r: CLEANERS.get(r["answer_format"], lambda s: str(s).strip())(r["answer_raw"]), axis=1)
    df["label_changed"] = df["answer"] != df["answer_raw"]
    report.append(f"rows with canonicalized answer: {int(df['label_changed'].sum())}")

    # open_ended leakage specifically
    leak = df["answer_raw"].str.contains(r"Errors?:", case=False, na=False, regex=True)
    report.append(f"open_ended 'Errors:' leakage rows cleaned: {int(leak.sum())}")

    dropped = []  # (row_index, reason)

    # 2. empty canonical answer
    empty = df["answer"].str.strip() == ""
    report.append(f"rows with empty canonical answer: {int(empty.sum())}")
    for i in df.index[empty]:
        dropped.append((i, "empty_answer"))

    # 3. exact content duplicates (after canonicalization)
    dup = df.duplicated(subset=["video", "timestamp_start", "question", "answer", "answer_format"], keep="first")
    report.append(f"exact duplicate rows (dropped): {int(dup.sum())}")
    for i in df.index[dup]:
        dropped.append((i, "duplicate"))

    # 4. label conflicts: same frame+question+format, different canonical gold
    grp = df.groupby(["video", "timestamp_start", "question", "answer_format"])["answer"].nunique()
    conflict_keys = grp[grp > 1].index
    conflict_mask = df.set_index(["video", "timestamp_start", "question", "answer_format"]).index.isin(conflict_keys)
    report.append(f"label-conflict prompt groups (dropped): {len(conflict_keys)}")
    report.append(f"rows in conflict groups (dropped): {int(conflict_mask.sum())}")
    for i in df.index[conflict_mask]:
        dropped.append((i, "label_conflict"))

    # 4b. cross-format duplication (kept, flagged only): same prompt appears under >1 format
    cross = df.groupby(["video", "timestamp_start", "question"])["answer_format"].nunique()
    report.append(f"cross-format prompt groups (kept, flagged): {int((cross > 1).sum())}")

    # 5. frame resolution
    if args.frames_manifest:
        frames = load_frames(args.frames_manifest)
        df["frame_path"] = df.apply(lambda r: resolve_frame(r["video"], r["timestamp_start"], frames), axis=1)
        missing = df["frame_path"].isna()
        report.append(f"rows with missing frame (dropped): {int(missing.sum())}")
        for i in df.index[missing]:
            dropped.append((i, "missing_frame"))
    else:
        df["frame_path"] = ""

    # apply drops (dedup reasons)
    drop_idx = set()
    for i, _reason in dropped:
        drop_idx.add(i)
    keep = [i for i in df.index if i not in drop_idx]
    df_clean = df.loc[keep].reset_index(drop=True)

    # serialize object/array metadata column so parquet round-trips cleanly
    if "secondary_capabilities" in df_clean.columns:
        df_clean["secondary_capabilities"] = df_clean["secondary_capabilities"].apply(
            lambda x: json.dumps(list(x)) if hasattr(x, "__iter__") and not isinstance(x, str) else x)

    report.append(f"\noutput rows: {len(df_clean)}  (removed {n0 - len(df_clean)})")

    # dropped file
    drop_rows = df.loc[sorted(drop_idx)]
    drop_rows["drop_reason"] = [dict(dropped)[i] for i in sorted(drop_idx)]
    drop_cols = ["id", "video", "procedure_type", "timestamp_start", "answer_format",
                 "answer_raw", "answer", "question", "drop_reason"]
    drop_rows[drop_cols].to_parquet(args.out.replace(".parquet", ".dropped.parquet"), index=False)

    df_clean.to_parquet(args.out, index=False)

    text = "\n".join(report)
    with open(args.out.replace(".parquet", ".report.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nwrote {args.out}")
    print(f"wrote {args.out.replace('.parquet', '.dropped.parquet')}")


if __name__ == "__main__":
    main()

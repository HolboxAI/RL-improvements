"""Verifiable reward for surgical-frame VQA (RLVR).

A deterministic, per-format reward ported from the swift reward plugin to a plain TRL
callable. This is the whole point of RLVR: a reward you can trust without a learned
reward model.

TRL calls a reward function as ``reward_func(completions, **kwargs)`` where
``completions`` is ``list[list[str]]`` (one inner list per prompt, one decoded string
per generation in that prompt's group) and ``**kwargs`` carries the remaining dataset
columns, each aligned to the prompts. We return a matching ``list[list[float]]``.

Scoring (matches CLAUDE.md §1):
    number          -> exact match of the first integer            (0.0 / 1.0)
    binary          -> exact match on yes/no                       (0.0 / 1.0)
    multiple_choice -> exact match                                 (0.0 / 1.0)
    fo_class        -> micro-F1 over the 8 canonical classes       (in [0, 1])
    open_ended      -> token-F1 (bag-of-words F1)                  (in [0, 1])

The <answer></answer> format gate is intentionally removed: the merged model answers
without the tags and the hard gate zeroed every reward. ``extract_answer`` still honors
tags when present and falls back to the raw completion otherwise.
"""
from __future__ import annotations

import re
from collections import Counter

FO_CLASSES = [
    "Clip", "Silicone loop", "Needle", "Sponge",
    "Specimen bag", "Specimen", "External drain", "Gallstone",
]

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(text: str) -> str:
    """Return the text between <answer> tags when present, else the raw completion."""
    if not text:
        return ""
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).strip(" .")


def _first_int(s: str):
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else None


def _parse_classes(s: str) -> set[str]:
    """Parse a multi-label answer like 'Clip, Sponge' into a normalized set."""
    if _normalize(s) in ("none", ""):
        return set()
    out = set()
    for part in re.split(r"[,;]|\band\b", s):
        part = _normalize(part)
        if part:
            out.add(part)
    return out


def _micro_f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    prec = tp / len(pred)
    rec = tp / len(gold)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _token_f1(pred: str, gold: str) -> float:
    p = Counter(pred.lower().split())
    g = Counter(gold.lower().split())
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = sum((p & g).values())
    prec = tp / sum(p.values())
    rec = tp / sum(g.values())
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def score(generated: str, gold: str, fmt: str) -> float:
    """Score a single completion against its gold answer for one format."""
    pred = extract_answer(generated)
    gold = (gold or "").strip()

    if fmt == "number":
        pi, gi = _first_int(pred), _first_int(gold)
        return 1.0 if (pi is not None and pi == gi) else 0.0
    if fmt in ("binary", "multiple_choice"):
        return 1.0 if _normalize(pred) == _normalize(gold) else 0.0
    if fmt == "fo_class":
        return _micro_f1(_parse_classes(pred), _parse_classes(gold))
    if fmt == "open_ended":
        return _token_f1(pred, gold)
    # unknown format: normalized exact match
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def reward_func(completions, answer, answer_format, **kwargs):
    """TRL reward entry point. Returns list[list[float]] aligned with completions."""
    rewards = []
    for comps, gold, fmt in zip(completions, answer, answer_format):
        rewards.append([score(c, gold, fmt) for c in comps])
    return rewards

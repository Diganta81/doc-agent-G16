"""Stage 9 — metrics"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
import unicodedata
from collections import Counter
from pathlib import Path

from ..contracts import *  # noqa
from ..llm.postprocess import support_ratio

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_DROP = re.compile(r"[\s.,;:!?()\[\]{}\"'“”‘’।॥\-–—/|]+")


def normalize(text: str) -> str:
    """NFC, ASCII digits, lower-case, no whitespace/punctuation — for exact-match comparison."""
    return _DROP.sub("", unicodedata.normalize("NFC", text or "").translate(_BN_DIGITS).lower())


def contains_gold(pred: str, gold: str) -> bool:
    """Verifiable-question check: the normalised gold answer occurs in the normalised prediction.

    A purely numeric gold ("80") must stand alone as a number: it does not match inside "800".
    """
    g = normalize(gold)
    if not g:
        return False
    if re.fullmatch(r"[0-9.,]+", g):
        text = unicodedata.normalize("NFC", pred or "").translate(_BN_DIGITS)
        return re.search(rf"(?<![0-9]){re.escape(g)}(?![0-9])", text) is not None
    return g in normalize(pred)


def gold_similarity(pred: str, gold: str) -> float:
    """Best character-level match (0-1) of the gold string anywhere in the prediction.

    1.0 = exact (same as contains_gold). Values around 0.7-0.9 are near-misses such as an OCR
    spelling variant of the right word; low values mean the gold answer is simply not there.
    """
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return 0.0
    if g in p:
        return 1.0
    return max(SequenceMatcher(None, p[i:i + len(g)], g).ratio()
               for i in range(max(1, len(p) - len(g) + 1)))


def ocr_f1(pred: str, gold: str) -> float:
    """Character-level F1 (multiset overlap) between OCR text and the labelled transcription."""
    p, g = Counter(normalize(pred)), Counter(normalize(gold))
    if not p or not g:
        return 0.0
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / sum(p.values()), overlap / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved: list, gold: list, k: int) -> float:
    """Share of gold ids found among the first k retrieved ids (1.0 when there is no gold)."""
    if not gold:
        return 1.0
    return len(set(gold) & set(retrieved[:k])) / len(set(gold))


def groundedness(answer: Answer, evidence: dict | None = None) -> float:
    """No-hallucination score in [0, 1].

    Without ``evidence`` (chunk_id -> text): 1.0 if the answer is grounded and cited, else 0.0.
    With it: the share of answer tokens found in the cited chunks (0.0 if it cites nothing).
    """
    if not answer.grounded or not answer.citations:
        return 0.0
    if evidence is None:
        return 1.0
    texts = [evidence[c.chunk_id] for c in answer.citations if c.chunk_id in evidence]
    return support_ratio(answer.text, texts) if texts else 0.0


def citation_accuracy(answer: Answer, chunks: dict | None = None) -> float:
    """Share of citations that point at a real chunk with an in-range span (1.0 if none to check).

    ``chunks`` maps chunk_id -> text. Without it only the span shape is checked.
    """
    if not answer.citations:
        return 1.0
    ok = 0
    for c in answer.citations:
        start, end = c.span
        if chunks is None:
            ok += 0 <= start < end
        else:
            ok += c.chunk_id in chunks and 0 <= start < end <= len(chunks[c.chunk_id])
    return ok / len(answer.citations)


def ece(confidences, correct, n_bins: int = 10) -> float:     # calibration
    """Expected calibration error with equal-width confidence bins."""
    pairs = list(zip(confidences, correct, strict=True))
    if not pairs:
        return 0.0
    total, error = len(pairs), 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        members = [(c, y) for c, y in pairs if lo < c <= hi or (b == 0 and c == 0.0)]
        if members:
            conf = sum(c for c, _ in members) / len(members)
            acc = sum(bool(y) for _, y in members) / len(members)
            error += len(members) / total * abs(acc - conf)
    return error


def subgroup_gap(scores_by_group: dict) -> float:
    """Largest difference between group mean scores (values: a score or a list of scores)."""
    means = [sum(v) / len(v) if isinstance(v, (list, tuple)) else float(v)
             for v in scores_by_group.values() if not isinstance(v, (list, tuple)) or v]
    return max(means) - min(means) if means else 0.0


def trace_completeness(path: str | Path = "traces/run.jsonl") -> dict:
    """Auditable-NFR check: share of runs whose trace can be replayed from the file alone.

    A run is complete when its steps are numbered 1..n without gaps, it starts with a retrieve call,
    every retrieve records query/k and the observed top_score/k, and it ends with an answer line.
    """
    runs: dict[str, list[dict]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            runs.setdefault(record.get("run_id", "?"), []).append(record)

    def complete(steps: list[dict]) -> bool:
        steps = sorted(steps, key=lambda s: s["step"])
        if [s["step"] for s in steps] != list(range(1, len(steps) + 1)):
            return False
        if steps[0]["tool"] != "retrieve" or steps[-1]["tool"] != "answer":
            return False
        for s in steps:
            if s["tool"] == "retrieve" and not (
                {"query", "k"} <= s["args"].keys() and {"top_score", "k"} <= s["obs"].keys()
            ):
                return False
            if s["tool"] == "answer" and "abstained" not in s["obs"]:
                return False
        return True

    done = sum(complete(steps) for steps in runs.values())
    return {"runs": len(runs), "complete": done, "rate": done / len(runs) if runs else 0.0}

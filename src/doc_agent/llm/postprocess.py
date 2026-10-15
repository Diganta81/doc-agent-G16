"""LLM — answer post-process / format / abstention"""
from __future__ import annotations

import re

from ..contracts import *  # noqa
from .prompts import ABSTAIN_TEXT

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
_MARKER = re.compile(r"\[C(\d+)\]")
# Bengali vowel signs are combining marks (not \w), so split on whitespace/punctuation instead.
_TOKEN = re.compile(r"[^\s.,;:!?()\[\]{}\"'“”‘’।॥\-–—/|<>]+")
_SENTENCE = re.compile(r"[^।.!?\n]+[।.!?]?")


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def support_ratio(answer_text: str, evidence_texts: list[str]) -> float:
    """Share of answer tokens (length >= 2) that also occur in the cited evidence."""
    answer_tokens = [t for t in tokens(answer_text) if len(t) >= 2]
    if not answer_tokens:
        return 0.0
    evidence_tokens = {t for text in evidence_texts for t in tokens(text)}
    return sum(1 for t in answer_tokens if t in evidence_tokens) / len(answer_tokens)


def best_span(claim: str, chunk_text: str) -> tuple[int, int]:
    """Character span of the chunk sentence that overlaps most with the claim."""
    claim_tokens = set(tokens(claim))
    best = (0, len(chunk_text))
    best_score = -1
    for match in _SENTENCE.finditer(chunk_text):
        if not match.group().strip():
            continue
        score = len(claim_tokens & set(tokens(match.group())))
        if score > best_score:
            best, best_score = (match.start(), match.end()), score
    return best


def evidence_chunks(state: dict) -> list[Chunk]:
    """Chunks the agent currently relies on: the latest retrieve/rerank observation."""
    for result in reversed(state.get("obs", [])):
        payload = getattr(result, "payload", None) or {}
        if getattr(result, "ok", False) and "chunks" in payload:
            return [Chunk.model_validate(c) for c in payload["chunks"]]
    return []


def evidence_sufficient(state: dict, cfg: dict) -> bool:
    chunks = evidence_chunks(state)
    return bool(chunks) and max(c.score for c in chunks) >= cfg["retrieve"]["weak_threshold"]


def parse_citations(raw: str, evidence: list[Chunk]) -> tuple[str, list[Citation]]:
    """Strip [C#] markers from ``raw`` and turn the valid ones into Citations."""
    citations: list[Citation] = []
    seen: set[tuple[str, int, int]] = set()
    cursor = 0
    for marker in _MARKER.finditer(raw):
        claim = raw[cursor:marker.start()]
        cursor = marker.end()
        index = int(marker.group(1)) - 1
        if not 0 <= index < len(evidence):
            continue
        chunk = evidence[index]
        start, end = best_span(claim, chunk.text)
        if (chunk.id, start, end) not in seen:
            seen.add((chunk.id, start, end))
            citations.append(Citation(chunk_id=chunk.id, span=(start, end)))
    clean = re.sub(r"\s+([।.,;])", r"\1", _MARKER.sub("", raw)).strip()
    return re.sub(r"[ \t]{2,}", " ", clean), citations


def abstain() -> Answer:
    return Answer(text=ABSTAIN_TEXT, citations=[], grounded=False, confidence=0.0)


def format_answer(
    raw: str,
    citations: list,
    evidence: list[Chunk] | None = None,
    top_score: float = 0.0,
    min_support: float = 0.3,
) -> Answer:
    """Attach citations, set grounded/confidence, enforce abstention.

    ``citations`` are validated Citation objects. When ``evidence`` is given the answer must also be
    lexically supported by the cited chunks (support_ratio >= min_support); otherwise it abstains.
    """
    text = (raw or "").strip()
    if not text or INSUFFICIENT in text or not citations:
        return abstain()

    support = 1.0
    if evidence is not None:
        by_id = {c.id: c.text for c in evidence}
        cited_texts = [by_id[c.chunk_id] for c in citations if c.chunk_id in by_id]
        if len(cited_texts) != len(citations):
            return abstain()  # citation points at a chunk that was never retrieved
        support = support_ratio(text, cited_texts)
        if support < min_support:
            return abstain()

    confidence = round(0.5 * support + 0.5 * min(1.0, max(0.0, top_score) / 0.7), 4)
    return Answer(text=text, citations=citations, grounded=True, confidence=confidence)


def register(hooks) -> None:
    """Wire the grounding / abstention gate (BEFORE_ANSWER).

    The handler records the evidence summary on the live agent state (audit). The abstention verdict
    itself is enforced in Agent.synthesize() via evidence_sufficient()/format_answer(): earlier
    BEFORE_ANSWER handlers (PII scrub) return copies of ctx, and the agent loop ignores the return
    value of that seam, so a verdict set only here could be lost.
    """

    def _ground(ctx: dict) -> dict:
        state = ctx.get("state")
        if isinstance(state, dict):
            chunks = evidence_chunks(state)
            state["grounding"] = {
                "n_evidence": len(chunks),
                "top_score": max((c.score for c in chunks), default=0.0),
            }
        return ctx

    hooks.register(hooks.BEFORE_ANSWER, _ground)

"""Per-task verifier. FIXED signature."""
from __future__ import annotations

import re
import unicodedata

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_DROP = re.compile(r"[\s.,;:!?()\[\]{}\"'“”‘’।॥\-–—/|]+")


def _norm(text: str) -> str:
    return _DROP.sub("", unicodedata.normalize("NFC", text or "").translate(_BN_DIGITS).lower())


def check(task: dict, answer: dict) -> bool:
    """Return True if `answer` satisfies `task` (exact for fact tasks; judge for open).

    - abstention task: the answer must be ungrounded with no citations (it declined to answer).
    - verifiable task: the answer is grounded, cited, and contains the normalised gold string.
    - judged task: scored by an LLM-judge / human, so only the grounding contract is checked here.
    """
    grounded = bool(answer.get("grounded")) and bool(answer.get("citations"))
    if task.get("abstain"):
        return not answer.get("grounded") and not answer.get("citations")
    if task.get("verifiable"):
        gold = _norm(task.get("gold", ""))
        text = answer.get("text", "")
        if gold and re.fullmatch(r"[0-9.,]+", gold):
            # a numeric gold must stand alone: "80" must not match inside "800"
            plain = unicodedata.normalize("NFC", text).translate(_BN_DIGITS)
            return grounded and re.search(rf"(?<![0-9]){re.escape(gold)}(?![0-9])", plain) is not None
        return grounded and bool(gold) and gold in _norm(text)
    return grounded

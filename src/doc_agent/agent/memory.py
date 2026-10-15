"""Stage 6 — working/episodic memory"""
from __future__ import annotations

import re

from ..contracts import *  # noqa

_TOKEN = re.compile(r"[^\s.,;:!?()\[\]{}\"'।॥\-/]+")


def _text_of(item) -> str:
    if isinstance(item, ToolResult):
        parts = [str(item.payload.get("query", ""))]
        parts += [c.get("text", "") for c in item.payload.get("chunks", [])]
        return " ".join(parts)
    return str(item)


class Memory:
    def __init__(self) -> None:
        self.items: list = []

    def add(self, item) -> None:
        self.items.append(item)

    def recall(self, query: str, limit: int = 3) -> list:
        """Items whose text shares the most tokens with ``query`` (most overlap first)."""
        wanted = set(_TOKEN.findall(query.lower()))
        scored = []
        for position, item in enumerate(self.items):
            overlap = len(wanted & set(_TOKEN.findall(_text_of(item).lower())))
            if overlap:
                scored.append((-overlap, -position, item))
        scored.sort(key=lambda row: row[:2])
        return [item for _, _, item in scored[:limit]]

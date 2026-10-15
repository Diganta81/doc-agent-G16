"""HITL — human-in-the-loop review queue"""
from __future__ import annotations

from ..contracts import *  # noqa
from . import hitl_store


def escalate(reason: str, context: dict) -> ToolResult:
    """Queue for human review. The caller must not act until the item is approved."""
    item_id = hitl_store.enqueue({"reason": reason, "context": context})
    return ToolResult(ok=True, payload={"review_id": item_id, "status": "pending", "reason": reason})


def review_queue():
    """Return pending items for the reviewer UI."""
    return hitl_store.pending()


def is_approved(review_id: str) -> bool:
    info = hitl_store.status(review_id)
    return info["status"] == "resolved" and info["decision"] == "approve"

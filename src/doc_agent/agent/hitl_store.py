"""HITL — persistent review queue (survives restarts)"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from ..contracts import *  # noqa

DB_PATH = Path("data/interim/hitl.sqlite")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS review (id TEXT PRIMARY KEY, created REAL, item TEXT, "
        "status TEXT, decision TEXT)"
    )
    return conn


def enqueue(item: dict) -> str:
    """Persist a pending review item; return its id."""
    item_id = uuid.uuid4().hex[:12]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO review VALUES (?, ?, ?, 'pending', NULL)",
            (item_id, time.time(), json.dumps(item, ensure_ascii=False, default=str)),
        )
    return item_id


def pending() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, created, item FROM review WHERE status = 'pending' ORDER BY created"
        ).fetchall()
    return [{"id": r[0], "created": r[1], **json.loads(r[2])} for r in rows]


def resolve(item_id: str, decision: str) -> None:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE review SET status = 'resolved', decision = ? WHERE id = ? AND status = 'pending'",
            (decision, item_id),
        )
    if cur.rowcount == 0:
        raise KeyError(f"no pending review item {item_id!r}")


def status(item_id: str) -> dict:
    with _connect() as conn:
        row = conn.execute(
            "SELECT status, decision FROM review WHERE id = ?", (item_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"unknown review item {item_id!r}")
    return {"status": row[0], "decision": row[1]}

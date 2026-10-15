"""Stage 6 — FIXED tool interface — the agent's tools"""
from __future__ import annotations

import ast
import operator
import re
import statistics

from abc import ABC, abstractmethod

from ..contracts import *  # noqa
from .. import hooks
from ..retrieval import rerank as rerank_mod
from ..retrieval import retriever as retriever_mod
from . import hitl

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def to_ascii_digits(text: str) -> str:
    return text.translate(_BN_DIGITS)


def _to_number(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(to_ascii_digits(str(value)).replace(",", ""))


class ToolContext:
    """Shared dependencies for the tools: config, retriever, and the chunk lookup."""

    def __init__(self, cfg: dict, retriever) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self._by_id: dict[str, Chunk] | None = None

    @property
    def chunks_by_id(self) -> dict[str, Chunk]:
        if self._by_id is None:
            self._by_id = {c.id: c for c in self.retriever.all_chunks()}
        return self._by_id


def _dump(chunk: Chunk) -> dict:
    return chunk.model_dump()


class Tool(ABC):
    name: str

    def __init__(self, ctx: ToolContext | None = None) -> None:
        self.ctx = ctx

    @abstractmethod
    def __call__(self, **kwargs) -> ToolResult: ...


def _fail(message: str) -> ToolResult:
    return ToolResult(ok=False, payload={"error": message})


# FIXED tool set — names & signatures locked (test_tools.py checks these).
class Retrieve(Tool):
    name = "retrieve"

    def __call__(self, query: str, k: int = 10) -> ToolResult:
        try:
            chunks = self.ctx.retriever.retrieve(query, k)
        except ValueError as exc:
            return _fail(str(exc))
        # Retrieved text is untrusted: let the security seam mask instruction-like spans.
        seam = hooks.run(hooks.AFTER_RETRIEVE, {"chunks": chunks})
        chunks = seam.get("chunks", chunks)
        return ToolResult(ok=True, payload={
            "query": query,
            "chunk_ids": [c.id for c in chunks],
            "chunks": [_dump(c) for c in chunks],
            "top_score": retriever_mod.top_score(chunks),
            "k": k,
            "n": len(chunks),
            "injection_flagged": seam.get("injection_flagged", 0),
        })


class Rerank(Tool):
    name = "rerank"

    def __call__(self, query: str, candidates: list) -> ToolResult:
        chunks: list[Chunk] = []
        for item in candidates:
            if isinstance(item, Chunk):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(Chunk.model_validate(item))
            elif item in self.ctx.chunks_by_id:
                chunks.append(self.ctx.chunks_by_id[item])
            else:
                return _fail(f"unknown candidate {item!r}")
        reranked, warning = chunks, None
        try:
            reranked = rerank_mod.rerank(query, chunks, self.ctx.cfg)
        except (ImportError, OSError) as exc:        # model unavailable -> keep dense order
            warning = str(exc)
        payload = {
            "query": query,
            "chunk_ids": [c.id for c in reranked],
            "chunks": [_dump(c) for c in reranked],
            "top_score": retriever_mod.top_score(reranked),   # dense score; threshold stays valid
            "k": len(reranked),
            "n": len(reranked),
            "reranked": warning is None and bool(self.ctx.cfg["retrieve"].get("rerank")),
        }
        if warning:
            payload["warning"] = warning
        return ToolResult(ok=True, payload=payload)


class ReadPage(Tool):
    name = "read_page"

    def __call__(self, page_id: str) -> ToolResult:
        found = [c for c in self.ctx.chunks_by_id.values() if page_id in c.page_ids]
        if not found:
            return _fail(f"no indexed text for page {page_id!r}")
        return ToolResult(ok=True, payload={
            "page_id": page_id,
            "chunk_ids": [c.id for c in found],
            "text": "\n".join(c.text for c in found),
        })


class EnhancePage(Tool):
    name = "enhance_page"

    def __call__(self, page_id: str) -> ToolResult:
        # Stage 1 enhancement is not part of this build (enhance.enabled is false).
        return _fail("page enhancement is disabled (enhance.enabled=false)")


class Extract(Tool):
    name = "extract"

    def __call__(self, field: str, chunk_id: str) -> ToolResult:
        chunk = self.ctx.chunks_by_id.get(chunk_id)
        if chunk is None:
            return _fail(f"unknown chunk {chunk_id!r}")
        text, key = chunk.text, field.strip().lower()
        if key in {"number", "numbers"}:
            raw = _NUMBER.findall(to_ascii_digits(text))
            values = [float(r.replace(",", "")) for r in raw]
        elif key == "percent":
            raw = re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|শতাংশ)", to_ascii_digits(text))
            values = [float(r) for r in raw]
        elif key == "year":
            raw = [y for y in re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", to_ascii_digits(text))]
            values = [float(y) for y in raw]
        else:   # any other field = keyword: return the sentences that mention it
            raw = [m.group().strip() for m in re.finditer(r"[^।.!?\n]+[।.!?]?", text)
                   if key in m.group().lower()]
            values = list(raw)
        return ToolResult(ok=True, payload={"field": field, "chunk_id": chunk_id,
                                            "values": values, "raw": raw})


class Aggregate(Tool):
    name = "aggregate"

    def __call__(self, op: str, items: list) -> ToolResult:
        if op == "count":
            return ToolResult(ok=True, payload={"op": op, "value": len(items)})
        if op == "unique":
            return ToolResult(ok=True, payload={"op": op, "value": sorted({str(i) for i in items})})
        try:
            numbers = [_to_number(i) for i in items]
        except ValueError:
            return _fail("aggregate needs numeric items")
        if not numbers:
            return _fail("aggregate needs at least one item")
        funcs = {"sum": sum, "mean": statistics.fmean, "min": min, "max": max}
        if op not in funcs:
            return _fail(f"unknown aggregate op {op!r}")
        return ToolResult(ok=True, payload={"op": op, "value": funcs[op](numbers)})


class Cite(Tool):
    name = "cite"

    def __call__(self, chunk_id: str, span: tuple) -> ToolResult:
        chunk = self.ctx.chunks_by_id.get(chunk_id)
        if chunk is None:
            return _fail(f"unknown chunk {chunk_id!r}")
        start, end = int(span[0]), int(span[1])
        if not 0 <= start < end <= len(chunk.text):
            return _fail(f"span {(start, end)} is outside chunk text of length {len(chunk.text)}")
        return ToolResult(ok=True, payload={
            "citation": Citation(chunk_id=chunk_id, span=(start, end)).model_dump(),
            "quote": chunk.text[start:end],
            "page_ids": chunk.page_ids,
        })


_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
            ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 64:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


class Calculator(Tool):
    name = "calculator"

    def __call__(self, expr: str) -> ToolResult:
        try:
            value = _safe_eval(ast.parse(to_ascii_digits(expr).strip(), mode="eval"))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
            return _fail(f"cannot evaluate {expr!r}: {exc}")
        return ToolResult(ok=True, payload={"expr": expr, "value": value})


class EscalateToHuman(Tool):     # HITL entry
    name = "escalate_to_human"

    def __call__(self, reason: str, context: dict) -> ToolResult:
        return hitl.escalate(reason, context)


REGISTRY = [Retrieve, Rerank, ReadPage, EnhancePage, Extract,
            Aggregate, Cite, Calculator, EscalateToHuman]


def build_tools(ctx: ToolContext) -> dict[str, Tool]:
    """Instantiate every registered tool against one shared context."""
    return {cls.name: cls(ctx) for cls in REGISTRY}

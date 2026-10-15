"""Stage 5 — reranking"""
from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa

_MODEL_CACHE: dict[str, Any] = {}


def _get_model(cfg: dict) -> Any:
    name = str(cfg["retrieve"]["reranker"])
    if name not in _MODEL_CACHE:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("Reranking requires sentence-transformers.") from exc
        device = str(cfg.get("retrieve", {}).get("rerank_device", "cpu"))
        _MODEL_CACHE[name] = CrossEncoder(name, device=device)
    return _MODEL_CACHE[name]


def rerank(query: str, candidates: list[Chunk], cfg: dict) -> list[Chunk]:
    """Cross-encoder rerank if cfg['retrieve']['rerank'].

    Only the ORDER changes. Each chunk keeps its dense retrieval score in ``Chunk.score`` because
    the weak-evidence threshold (retriever.is_weak) is calibrated on dense cosine scores, not on
    cross-encoder logits.
    """
    if not cfg["retrieve"].get("rerank") or len(candidates) < 2:
        return list(candidates)

    model = _get_model(cfg)
    pairs = [(query, chunk.text) for chunk in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
    return [candidates[i] for i in order]

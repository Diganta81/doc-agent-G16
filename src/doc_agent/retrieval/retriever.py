"""Stage 5 - dense retrieval over the persisted FAISS index."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import Chunk
from ..index import embed, store


class Retriever:
    """Retrieve ranked chunks using BGE-M3 and the persisted FAISS index."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.retrieve_cfg = cfg.get("retrieve", {})
        self._index: Any | None = None
        self._chunks: list[Chunk] | None = None

    def _load(self) -> tuple[Any, list[Chunk]]:
        if self._index is None or self._chunks is None:
            self._index, self._chunks = store.load(self.cfg)
        return self._index, self._chunks

    def all_chunks(self) -> list[Chunk]:
        """Every indexed chunk (for page reads, citation checks and id lookups)."""
        return list(self._load()[1])

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Return the highest-scoring chunks for a text query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        requested_k = int(self.retrieve_cfg.get("k", 10) if k is None else k)
        if requested_k <= 0:
            raise ValueError("k must be positive")

        index, chunks = self._load()
        if index.ntotal == 0:
            return []

        query_chunk = Chunk(
            id="query",
            doc_id="query",
            text=query,
            page_ids=[],
        )
        query_vector = embed.encode([query_chunk], self.cfg)
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.shape[1] != index.d:
            raise ValueError(
                f"Query dimension {query_vector.shape[1]} does not match index dimension {index.d}"
            )

        scores, ids = index.search(query_vector, min(requested_k, index.ntotal))
        results: list[Chunk] = []
        for score, chunk_id in zip(scores[0], ids[0], strict=True):
            if chunk_id < 0:
                continue
            results.append(chunks[int(chunk_id)].model_copy(update={"score": float(score)}))
        return results


# --- evidence-strength policy: read by agent.decide() for evidence-gated re-search ---
def top_score(chunks: list[Chunk]) -> float:
    """Strength of the current evidence = best chunk score (0.0 if empty)."""
    return max((chunk.score for chunk in chunks), default=0.0)


def is_weak(chunks: list[Chunk], cfg: dict) -> bool:
    """Weak evidence = best score below cfg.retrieve.weak_threshold."""
    return top_score(chunks) < cfg["retrieve"]["weak_threshold"]


def next_k(k: int, cfg: dict) -> int | None:
    """Widen retrieval until k_max, then signal that evidence is insufficient."""
    retrieve_cfg = cfg["retrieve"]
    next_value = k + int(retrieve_cfg["k_step"])
    return next_value if next_value <= int(retrieve_cfg["k_max"]) else None

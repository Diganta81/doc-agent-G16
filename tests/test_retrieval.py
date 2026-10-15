"""Unit tests for retrieval policy and reranking (no model downloads)."""
from doc_agent.contracts import Chunk
from doc_agent.retrieval import rerank as rerank_mod
from doc_agent.retrieval import retriever


def chunks(*scores):
    return [Chunk(id=f"c{i}", doc_id="d", text=f"t{i}", page_ids=["p"], score=s)
            for i, s in enumerate(scores)]


def test_top_score_and_is_weak(cfg):
    assert retriever.top_score([]) == 0.0
    assert retriever.is_weak(chunks(0.2, 0.1), cfg)
    assert not retriever.is_weak(chunks(0.2, 0.9), cfg)


def test_next_k_widens_then_signals_exhaustion(cfg):
    assert retriever.next_k(10, cfg) == 20
    assert retriever.next_k(30, cfg) == 40
    assert retriever.next_k(40, cfg) is None


class FakeCE:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, show_progress_bar=False):
        return self.scores


def test_rerank_orders_by_model_but_keeps_dense_scores(cfg, monkeypatch):
    cfg["retrieve"]["rerank"] = True
    monkeypatch.setitem(rerank_mod._MODEL_CACHE, "fake/cross-encoder", FakeCE([0.1, 0.9, 0.5]))
    out = rerank_mod.rerank("q", chunks(0.8, 0.7, 0.6), cfg)
    assert [c.id for c in out] == ["c1", "c2", "c0"]
    assert [c.score for c in out] == [0.7, 0.6, 0.8]      # dense scores untouched


def test_rerank_off_returns_input_order(cfg):
    cfg["retrieve"]["rerank"] = False
    assert [c.id for c in rerank_mod.rerank("q", chunks(0.1, 0.9), cfg)] == ["c0", "c1"]

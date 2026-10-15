"""Shared fixtures: a tiny config, a scripted fake retriever, isolated trace/HITL paths."""
import copy

import pytest

from doc_agent import hooks, logging_conf
from doc_agent.agent import hitl_store
from doc_agent.contracts import Chunk

CFG = {
    "seed": 42,
    "retrieve": {"k": 10, "k_step": 10, "k_max": 40, "weak_threshold": 0.35, "rerank": False,
                 "reranker": "fake/cross-encoder"},
    "agent": {"max_steps": 8, "autonomy": "act-with-approval", "budget_usd": 0.05,
              "cost_per_call_usd": 0.0, "research": True, "reformulate": True,
              "evidence_chunks": 5, "min_support": 0.3},
    "llm": {"provider": "none"},
}

TEXTS = [
    "ব্যাকটেরিয়া এককোষী জীব। ইহারা দ্বি-ভাজন পদ্ধতিতে প্রজনন করে।",
    "ভাইরাস রোগে পাতায় মোজাইক লক্ষণ দেখা যায়। রোগাক্রান্ত পাতার রস সংক্রমণ ঘটায়।",
    "ছত্রাকনাশক গাছকে ছত্রাকের আক্রমণ হইতে রক্ষা করে। বোর্দোমিশ্চার তামাযুক্ত ছত্রাকনাশক।",
    "ধানের ব্লাস্ট রোগ Pyricularia oryzae ছত্রাকের কারণে হয়।",
    "মহামারী সৃষ্টিতে আবহাওয়া ও সংবেদনশীল উদ্ভিদের প্রাচুর্য ভূমিকা রাখে।",
] * 9   # 45 chunks so k up to 40 is meaningful


def make_chunks() -> list[Chunk]:
    return [Chunk(id=f"c{i}", doc_id="d1", text=t, page_ids=[f"page-{35 + i:04d}"])
            for i, t in enumerate(TEXTS)]


class FakeRetriever:
    """retrieve(query, k) returns k chunks; the best score is top_score_fn(query, k)."""

    def __init__(self, top_score_fn):
        self.top_score_fn = top_score_fn
        self.chunks = make_chunks()
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query, k=None):
        self.calls.append((query, k))
        top = self.top_score_fn(query, k)
        return [c.model_copy(update={"score": top * (0.95 ** i)}) for i, c in enumerate(self.chunks[:k])]

    def all_chunks(self):
        return list(self.chunks)


@pytest.fixture
def cfg():
    return copy.deepcopy(CFG)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Trace file and HITL database in tmp_path; hooks cleared before and after."""
    monkeypatch.setattr(logging_conf, "TRACE_PATH", tmp_path / "run.jsonl")
    monkeypatch.setattr(hitl_store, "DB_PATH", tmp_path / "hitl.sqlite")
    hooks.clear()
    yield tmp_path
    hooks.clear()

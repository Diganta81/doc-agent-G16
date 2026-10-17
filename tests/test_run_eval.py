"""scripts/run_eval.py: the LLM guard rails (no silent extractive fallback)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_eval  # noqa: E402

from doc_agent.llm.client import LLM, LLMUnavailable  # noqa: E402

from .conftest import FakeRetriever  # noqa: E402

TASKS = [
    {"id": "t1", "question": "ব্যাকটেরিয়া কীভাবে প্রজনন করে?", "verifiable": False, "judged": True,
     "gold": "", "gold_pages": [], "needs_research": False},
    {"id": "t2", "question": "ভাইরাস রোগের লক্ষণ কী?", "verifiable": False, "judged": True,
     "gold": "", "gold_pages": [], "needs_research": False},
]


def _with_transport(monkeypatch, transport):
    monkeypatch.setattr(run_eval, "LLM", lambda cfg: LLM(cfg, transport=transport))


def _llm_cfg(cfg):
    cfg["llm"] = {"provider": "openai_compat", "base_url": "http://x", "model": "m"}
    return cfg


def test_run_stops_at_the_first_failed_llm_call(cfg, isolated, monkeypatch):
    def quota(url, headers, body):
        raise LLMUnavailable("HTTP 429 quota")

    _with_transport(monkeypatch, quota)
    with pytest.raises(run_eval.LLMStopped, match="429"):
        run_eval.evaluate(_llm_cfg(cfg), TASKS, FakeRetriever(lambda q, k: 0.8), isolated / "t.jsonl",
                          verbose=False)


def test_tolerating_failures_is_explicit_and_reported(cfg, isolated, monkeypatch):
    def quota(url, headers, body):
        raise LLMUnavailable("HTTP 429 quota")

    _with_transport(monkeypatch, quota)
    result = run_eval.evaluate(_llm_cfg(cfg), TASKS, FakeRetriever(lambda q, k: 0.8), isolated / "t.jsonl",
                               verbose=False, max_llm_failures=10)
    assert result["summary"]["llm_failures"] >= 2 and result["summary"]["tasks_with_llm_failure"] == 2
    assert all(row["llm_failed"] for row in result["rows"])


def test_healthy_llm_run_reports_calls_and_no_failures(cfg, isolated, monkeypatch):
    _with_transport(monkeypatch, lambda *a: {"choices": [{"message": {"content": "ইহারা দ্বি-ভাজন পদ্ধতিতে প্রজনন করে। [C1]"}}]})
    result = run_eval.evaluate(_llm_cfg(cfg), TASKS[:1], FakeRetriever(lambda q, k: 0.8),
                               isolated / "t.jsonl", verbose=False)
    summary = result["summary"]
    assert summary["llm_failures"] == 0 and summary["llm_calls"] >= 1
    assert summary["trace_complete_rate"] == 1.0

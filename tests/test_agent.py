"""Agent unit tests: evidence-gated re-search, abstention, grounding, guardrails, HITL."""
import json

from doc_agent import wiring
from doc_agent.agent import hitl, hitl_store
from doc_agent.agent.agent import Agent
from doc_agent.agent.memory import Memory
from doc_agent.contracts import ToolResult
from doc_agent.llm.client import LLM

from .conftest import FakeRetriever

QUESTION = "ব্যাকটেরিয়া কীভাবে প্রজনন করে?"


def trace_lines(tmp_path):
    return [json.loads(line) for line in (tmp_path / "run.jsonl").read_text().splitlines()]


def run_agent(cfg, retriever, llm=None, question=QUESTION):
    wiring.register_all(cfg)
    return Agent(cfg, retriever, llm=llm).answer_safely(question)


def scripted_llm(cfg, rewrite="ব্যাকটেরিয়ার প্রজনন পদ্ধতি", answer="ইহারা দ্বি-ভাজন পদ্ধতিতে প্রজনন করে। [C1]"):
    cfg["llm"] = {"provider": "openai_compat", "base_url": "http://x", "model": "m"}

    def transport(url, headers, body):
        prompt = body["messages"][0]["content"]
        text = rewrite if "Rewrite the question" in prompt else answer
        return {"choices": [{"message": {"content": text}}]}

    return LLM(cfg, transport=transport)


def test_control_single_pass(cfg, isolated):
    retriever = FakeRetriever(lambda q, k: 0.8)
    ans = run_agent(cfg, retriever)
    assert [k for _, k in retriever.calls] == [10]                 # one pass, k never widened
    assert ans.grounded and len(ans.citations) >= 1
    steps = trace_lines(isolated)
    assert [s["tool"] for s in steps] == ["retrieve", "answer"]
    assert steps[0]["obs"]["top_score"] == 0.8 and steps[0]["obs"]["k"] == 10


def test_trigger_widens_k_then_abstains(cfg, isolated):
    retriever = FakeRetriever(lambda q, k: 0.2)                    # always below weak_threshold 0.35
    ans = run_agent(cfg, retriever)
    assert [k for _, k in retriever.calls] == [10, 20, 30, 40]     # widened by k_step up to k_max
    assert not ans.grounded and ans.citations == []                # abstained at k_max
    steps = trace_lines(isolated)
    assert [s["obs"].get("k") for s in steps if s["tool"] == "retrieve"] == [10, 20, 30, 40]
    assert steps[-1]["tool"] == "answer" and steps[-1]["obs"]["abstained"] is True
    assert steps[-1]["obs"]["stop_reason"] == "abstain_weak_evidence"


def test_trigger_recovers_after_widening_and_reformulation(cfg, isolated):
    cfg["retrieve"]["k_max"] = 30
    # weak for the user's wording, strong once the agent rewrites the query and widens k
    retriever = FakeRetriever(lambda q, k: 0.8 if (q != QUESTION and k >= 20) else 0.2)
    ans = run_agent(cfg, retriever, llm=scripted_llm(cfg))
    assert [k for _, k in retriever.calls] == [10, 20]
    assert retriever.calls[1][0] == "ব্যাকটেরিয়ার প্রজনন পদ্ধতি"       # reformulated query was used
    assert ans.grounded and ans.citations
    assert [s["tool"] for s in trace_lines(isolated)] == ["retrieve", "retrieve", "answer"]


def test_path_depends_on_observations(cfg, isolated):
    strong, weak = FakeRetriever(lambda q, k: 0.8), FakeRetriever(lambda q, k: 0.2)
    run_agent(cfg, strong); run_agent(cfg, weak)
    assert len(strong.calls) == 1 and len(weak.calls) == 4


def test_research_ablation_switch_gives_single_pass(cfg, isolated):
    cfg["agent"]["research"] = False
    retriever = FakeRetriever(lambda q, k: 0.2)
    ans = run_agent(cfg, retriever)
    assert len(retriever.calls) == 1 and not ans.grounded


def test_rerank_step_runs_when_enabled(cfg, isolated, monkeypatch):
    from doc_agent.retrieval import rerank as rerank_mod

    class FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            return [float(i) for i in range(len(pairs))]          # reverses the order
    monkeypatch.setitem(rerank_mod._MODEL_CACHE, "fake/cross-encoder", FakeCE())
    cfg["retrieve"]["rerank"] = True
    ans = run_agent(cfg, FakeRetriever(lambda q, k: 0.8))
    assert [s["tool"] for s in trace_lines(isolated)] == ["retrieve", "rerank", "answer"]
    assert ans.grounded


def test_llm_answer_with_unsupported_claim_abstains(cfg, isolated):
    llm = scripted_llm(cfg, answer="কলার ভিটামিন সি মাটির আর্দ্রতা বৃষ্টিপাত সার কীটনাশক। [C1]")
    ans = run_agent(cfg, FakeRetriever(lambda q, k: 0.8), llm=llm)
    assert not ans.grounded and ans.citations == []


def test_llm_insufficient_evidence_abstains(cfg, isolated):
    llm = scripted_llm(cfg, answer="INSUFFICIENT_EVIDENCE")
    ans = run_agent(cfg, FakeRetriever(lambda q, k: 0.8), llm=llm)
    assert not ans.grounded


def test_citation_span_points_at_supporting_sentence(cfg, isolated):
    ans = run_agent(cfg, FakeRetriever(lambda q, k: 0.8))
    chunk = FakeRetriever(lambda q, k: 0).chunks[0]
    start, end = ans.citations[0].span
    assert "প্রজনন" in chunk.text[start:end]


def test_budget_exceeded_is_blocked_and_audited(cfg, isolated):
    cfg["agent"]["cost_per_call_usd"] = 0.03                         # 2nd call exceeds 0.05
    retriever = FakeRetriever(lambda q, k: 0.2)
    ans = run_agent(cfg, retriever)
    assert not ans.grounded and len(retriever.calls) == 1
    steps = trace_lines(isolated)
    assert steps[-1]["tool"] == "answer" and steps[-1]["obs"]["blocked"] == "cost_budget"


def test_state_changing_tool_needs_approval_and_is_queued(cfg, isolated):
    from doc_agent.agent.guardrails import Guardrails, GuardrailViolation
    import pytest
    g = Guardrails(cfg)
    with pytest.raises(GuardrailViolation) as info:
        g.check({"tool": "enhance_page", "args": {"page_id": "page-0035"}})
    assert info.value.kind == "approval_required"
    g.check({"tool": "enhance_page", "args": {"page_id": "page-0035"}, "approved": True})


def test_hitl_queue_survives_reconnect(isolated):
    item = hitl.escalate("low confidence", {"query": "q"})
    assert item.payload["status"] == "pending"
    pending = hitl.review_queue()
    assert len(pending) == 1 and pending[0]["reason"] == "low confidence"
    assert not hitl.is_approved(item.payload["review_id"])
    hitl_store.resolve(item.payload["review_id"], "approve")
    assert hitl.is_approved(item.payload["review_id"]) and hitl.review_queue() == []


def test_memory_recall_ranks_by_overlap():
    mem = Memory()
    mem.add(ToolResult(ok=True, payload={"query": "ধানের ব্লাস্ট", "chunks": []}))
    mem.add(ToolResult(ok=True, payload={"query": "ভাইরাস মোজাইক", "chunks": []}))
    assert mem.recall("ভাইরাস রোগ")[0].payload["query"] == "ভাইরাস মোজাইক"
    assert mem.recall("সম্পর্কহীন") == []

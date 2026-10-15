"""LLM client and answer post-processing."""
import pytest

from doc_agent.llm import postprocess
from doc_agent.llm.client import LLM, LLMUnavailable

from .conftest import make_chunks


def test_provider_none_is_unavailable(cfg):
    llm = LLM(cfg)
    assert not llm.available
    with pytest.raises(LLMUnavailable):
        llm.complete("hi")


def test_complete_uses_transport_and_returns_text(cfg):
    cfg["llm"] = {"provider": "openai_compat", "base_url": "http://x/v1/", "model": "m"}
    seen = {}

    def transport(url, headers, body):
        seen.update(url=url, body=body)
        return {"choices": [{"message": {"content": "  উত্তর  "}}]}

    llm = LLM(cfg, transport=transport)
    assert llm.complete("প্রশ্ন", max_tokens=7) == "উত্তর"
    assert seen["url"] == "http://x/v1/chat/completions" and seen["body"]["max_tokens"] == 7


def test_malformed_response_raises_unavailable(cfg):
    cfg["llm"] = {"provider": "openai_compat", "base_url": "http://x", "model": "m"}
    with pytest.raises(LLMUnavailable):
        LLM(cfg, transport=lambda *a: {"oops": 1}).complete("x")


def test_parse_citations_ignores_invalid_markers():
    evidence = [c.model_copy(update={"score": 0.9}) for c in make_chunks()[:2]]
    text, cites = postprocess.parse_citations("ইহারা এককোষী জীব। [C1] অজানা। [C9]", evidence)
    assert "[C" not in text and [c.chunk_id for c in cites] == ["c0"]


def test_tokens_keep_bengali_vowel_signs():
    assert postprocess.tokens("ব্যাকটেরিয়া, দ্বি-ভাজন।") == ["ব্যাকটেরিয়া", "দ্বি", "ভাজন"]


def test_format_answer_rules():
    chunks = make_chunks()[:1]
    cite = [postprocess.Citation(chunk_id="c0", span=(0, 20))]
    ok = postprocess.format_answer("ব্যাকটেরিয়া এককোষী জীব।", cite, evidence=chunks, top_score=0.7)
    assert ok.grounded and 0 < ok.confidence <= 1
    assert not postprocess.format_answer("INSUFFICIENT_EVIDENCE", cite).grounded
    assert not postprocess.format_answer("কিছু একটা", []).grounded                  # no citation
    ghost = [postprocess.Citation(chunk_id="zz", span=(0, 3))]
    assert not postprocess.format_answer("ব্যাকটেরিয়া", ghost, evidence=chunks).grounded


def test_pipe_and_danda_both_end_a_sentence():
    text = "প্রথম বাক্য | দ্বিতীয় বাক্য। তৃতীয় বাক্য?"
    parts = [m.group().strip() for m in postprocess._SENTENCE.finditer(text)]
    assert parts == ["প্রথম বাক্য |", "দ্বিতীয় বাক্য।", "তৃতীয় বাক্য?"]


def test_failures_are_counted_so_eval_can_show_silent_fallbacks(cfg):
    cfg["llm"] = {"provider": "openai_compat", "base_url": "http://x", "model": "m"}

    def boom(*args):
        raise LLMUnavailable("HTTP 429 quota")

    llm = LLM(cfg, transport=boom)
    with pytest.raises(LLMUnavailable):
        llm.complete("x")
    assert llm.calls == 1 and llm.failures == 1 and "429" in llm.last_error

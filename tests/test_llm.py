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


def test_min_interval_spaces_calls(cfg):
    import time
    cfg["llm"] = {"provider": "openai_compat", "base_url": "http://x", "model": "m", "min_interval_s": 0.15}
    llm = LLM(cfg, transport=lambda *a: {"choices": [{"message": {"content": "ok"}}]})
    started = time.monotonic()
    llm.complete("a"); llm.complete("b")
    assert time.monotonic() - started >= 0.14


def test_key_with_stray_whitespace_is_stripped(cfg, monkeypatch):
    from doc_agent.llm import client
    cfg["llm"] = {"provider": "openai_compat", "base_url": "http://x", "model": "m"}
    seen = {}
    monkeypatch.setattr(client.settings, "llm_api_key", "  secret-key\n")
    llm = LLM(cfg, transport=lambda url, headers, body: seen.update(h=headers) or
              {"choices": [{"message": {"content": "ok"}}]})
    assert llm.complete("x") == "ok" and seen["h"]["Authorization"] == "Bearer secret-key"


def test_whitespace_only_key_counts_as_missing(cfg, monkeypatch):
    from doc_agent.llm import client
    cfg["llm"] = {"provider": "openai_compat", "base_url": "http://x", "model": "m"}
    monkeypatch.setattr(client.settings, "llm_api_key", " \n")
    assert not LLM(cfg).available


def _counting_llm(cfg, tmp_path, replies):
    cfg["llm"] = {"provider": "openai_compat", "base_url": "http://x", "model": "m",
                  "cache_path": str(tmp_path / "cache.jsonl")}
    seen = []

    def transport(url, headers, body):
        seen.append(body)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"choices": [{"message": {"content": reply}}]}

    return LLM(cfg, transport=transport), seen


def test_cache_skips_repeated_prompts_and_survives_a_new_client(cfg, tmp_path):
    first, seen = _counting_llm(cfg, tmp_path, ["উত্তর"])
    assert first.complete("প্রশ্ন") == "উত্তর" and first.complete("প্রশ্ন") == "উত্তর"
    assert len(seen) == 1 and first.cache_hits == 1 and first.calls == 1
    second, seen2 = _counting_llm(cfg, tmp_path, [])            # new process, same cache file
    assert second.complete("প্রশ্ন") == "উত্তর" and seen2 == []


def test_cache_key_covers_the_prompt(cfg, tmp_path):
    llm, seen = _counting_llm(cfg, tmp_path, ["এক", "দুই"])
    assert llm.complete("a") == "এক" and llm.complete("b") == "দুই" and len(seen) == 2


def test_failures_are_not_cached_so_a_rerun_retries(cfg, tmp_path):
    llm, _ = _counting_llm(cfg, tmp_path, [LLMUnavailable("HTTP 429"), "ঠিক আছে"])
    with pytest.raises(LLMUnavailable):
        llm.complete("p")
    assert llm.complete("p") == "ঠিক আছে" and llm.failures == 1

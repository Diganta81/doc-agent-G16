from doc_agent.agent import tools
from doc_agent.agent.tools import ToolContext, build_tools

from .conftest import FakeRetriever


def test_registry_is_tool_subclasses():
    for t in tools.REGISTRY:
        assert issubclass(t, tools.Tool) and isinstance(t.name, str)


def make_tools(cfg):
    return build_tools(ToolContext(cfg, FakeRetriever(lambda q, k: 0.8)))


def test_retrieve_payload_carries_evidence_strength(cfg, isolated):
    result = make_tools(cfg)["retrieve"](query="ভাইরাস", k=3)
    assert result.ok and result.payload["k"] == 3 and result.payload["n"] == 3
    assert result.payload["top_score"] == 0.8 and len(result.payload["chunk_ids"]) == 3


def test_retrieve_blank_query_fails_softly(cfg):
    class Strict(FakeRetriever):
        def retrieve(self, query, k=None):
            raise ValueError("query must be a non-empty string")
    result = build_tools(ToolContext(cfg, Strict(lambda q, k: 0)))["retrieve"](query=" ", k=3)
    assert not result.ok and "non-empty" in result.payload["error"]


def test_read_page_and_unknown_page(cfg):
    t = make_tools(cfg)
    assert t["read_page"](page_id="page-0035").ok
    assert not t["read_page"](page_id="page-9999").ok


def test_extract_numbers_handles_bengali_digits(cfg):
    ctx = ToolContext(cfg, FakeRetriever(lambda q, k: 0))
    chunk = ctx.chunks_by_id["c0"].model_copy(update={"text": "৮০ ডিগ্রি এবং ২.৫ গ্রাম, ১৮৯২ সালে"})
    ctx._by_id = {"c0": chunk}
    t = build_tools(ctx)
    assert t["extract"](field="numbers", chunk_id="c0").payload["values"] == [80.0, 2.5, 1892.0]
    assert t["extract"](field="year", chunk_id="c0").payload["values"] == [1892.0]
    assert not t["extract"](field="numbers", chunk_id="nope").ok


def test_extract_keyword_returns_matching_sentences(cfg):
    out = make_tools(cfg)["extract"](field="প্রজনন", chunk_id="c0")
    assert out.ok and any("প্রজনন" in s for s in out.payload["values"])


def test_aggregate_ops(cfg):
    agg = make_tools(cfg)["aggregate"]
    assert agg(op="sum", items=["১০", 5]).payload["value"] == 15
    assert agg(op="max", items=[1, 9, 3]).payload["value"] == 9
    assert agg(op="count", items=[1, 2]).payload["value"] == 2
    assert not agg(op="median", items=[1]).ok and not agg(op="sum", items=["x"]).ok


def test_cite_validates_span(cfg):
    cite = make_tools(cfg)["cite"]
    ok = cite(chunk_id="c0", span=(0, 5))
    assert ok.ok and ok.payload["quote"]
    assert not cite(chunk_id="c0", span=(0, 10_000)).ok
    assert not cite(chunk_id="nope", span=(0, 1)).ok


def test_calculator_is_safe(cfg):
    calc = make_tools(cfg)["calculator"]
    assert calc(expr="২ * (3 + 4)").payload["value"] == 14
    assert not calc(expr="__import__('os').system('echo hi')").ok
    assert not calc(expr="2 ** 9999").ok
    assert not calc(expr="1 / 0").ok


def test_enhance_page_is_disabled_not_faked(cfg):
    assert not make_tools(cfg)["enhance_page"](page_id="page-0035").ok


def test_escalate_tool_queues_review(cfg, isolated):
    out = make_tools(cfg)["escalate_to_human"](reason="needs expert", context={"q": 1})
    assert out.ok and out.payload["status"] == "pending"

"""Run tasks.jsonl through the agent and score.

    python scripts/run_eval.py                         # score the task suite -> reports/eval_results.json
    python scripts/run_eval.py --ablate                # also toggle rerank / re-search / reformulation off
    python scripts/run_eval.py --set embed.device=cuda --set retrieve.reranker=BAAI/bge-reranker-v2-m3

Needs a built index (data/index/index.faiss, chunks.jsonl, metadata.json) - see scripts/run_index.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import yaml

from doc_agent import config, logging_conf, wiring
from doc_agent.agent.agent import Agent
from doc_agent.agent.tools import ToolContext, build_tools
from doc_agent.eval import ablation, metrics
from doc_agent.llm.client import LLM
from doc_agent.retrieval.retriever import Retriever

ROOT = Path(__file__).resolve().parent.parent


def _load_checker():
    spec = importlib.util.spec_from_file_location("success_check", ROOT / "grading_kit/success_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """--set a.b.c=value (value parsed as YAML so numbers/bools/lists work)."""
    for item in overrides:
        dotted, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def load_tasks(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return [r for r in rows if r.get("question")]


class RecordingRetriever:
    """Wraps a Retriever and records every call (query, k, top score, ranked page ids)."""

    def __init__(self, inner: Retriever) -> None:
        self.inner, self.calls = inner, []

    def retrieve(self, query: str, k: int | None = None):
        chunks = self.inner.retrieve(query, k)
        self.calls.append({"query": query, "k": k, "top_score": max((c.score for c in chunks), default=0.0),
                           "pages": [p for c in chunks for p in c.page_ids]})
        return chunks

    def all_chunks(self):
        return self.inner.all_chunks()


def _base(page_id: str) -> str:
    return page_id.rsplit("/", 1)[-1]


def _dedupe(seq: list[str]) -> list[str]:
    seen, out = set(), []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def evaluate(cfg: dict, tasks: list[dict], retriever, trace_path: Path, verbose: bool = True) -> dict:
    """Run every task once. Returns {"summary": {...numbers...}, "rows": [...]}."""
    logging_conf.TRACE_PATH = trace_path
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()
    check = _load_checker()
    llm = LLM(cfg)
    chunk_text = {c.id: c.text for c in retriever.all_chunks()}
    chunk_pages = {c.id: c.page_ids for c in retriever.all_chunks()}
    rows = []
    for task in tasks:
        wiring.register_all(cfg)                                   # fresh guardrail budget per query
        recorder = RecordingRetriever(retriever)
        agent = Agent(cfg, recorder, llm=llm, tools=build_tools(ToolContext(cfg, recorder)))
        failures_before = llm.failures
        started = time.perf_counter()
        answer = agent.answer_safely(task["question"])
        latency = time.perf_counter() - started
        gold = [_base(p) for p in task.get("gold_pages", [])]
        first = recorder.calls[0] if recorder.calls else {"top_score": 0.0, "pages": []}
        ranked_first = [_base(p) for p in _dedupe(first["pages"])]
        cited_pages = {_base(p) for c in answer.citations for p in chunk_pages.get(c.chunk_id, [])}
        row = {
            "id": task["id"], "question": task["question"], "answer": answer.text,
            "grounded": answer.grounded, "confidence": answer.confidence,
            "n_citations": len(answer.citations),
            "ok": bool(check(task, answer.model_dump())),
            "verifiable": bool(task.get("verifiable")), "judged": bool(task.get("judged")),
            "needs_research": task.get("needs_research", False), "abstain_task": task.get("abstain", False),
            "k_path": [c["k"] for c in recorder.calls],
            "top_score_path": [round(c["top_score"], 4) for c in recorder.calls],
            "queries": [c["query"] for c in recorder.calls],
            "researched": len(recorder.calls) > 1,
            "recall_at_10": metrics.recall_at_k(ranked_first, gold, 10) if gold else None,
            "recall_at_5": metrics.recall_at_k(ranked_first, gold, 5) if gold else None,
            "top1_page_hit": bool(ranked_first and ranked_first[0] in gold) if gold else None,
            "cited_gold_page": bool(cited_pages & set(gold)) if gold else None,
            "groundedness": metrics.groundedness(answer, chunk_text),
            "citation_accuracy": metrics.citation_accuracy(answer, chunk_text),
            "latency_s": round(latency, 3),
            "llm_failed": llm.failures > failures_before,   # True = this answer may be an extractive fallback
        }
        rows.append(row)
        if verbose:
            print(f"[{row['id']}] ok={row['ok']} grounded={row['grounded']} k={row['k_path']} "
                  f"top={row['top_score_path']}", flush=True)
    summary = summarize(rows, trace_path)
    summary.update(llm_provider=llm.provider, llm_calls=llm.calls, llm_failures=llm.failures,
                   tasks_with_llm_failure=sum(r["llm_failed"] for r in rows))
    if llm.failures:
        summary["llm_last_error"] = llm.last_error        # failures silently fall back to extractive
    return {"summary": summary, "rows": rows}


def _mean(values: list) -> float:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else 0.0


def summarize(rows: list[dict], trace_path: Path) -> dict:
    answerable = [r for r in rows if not r["abstain_task"]]
    verifiable = [r for r in rows if r["verifiable"]]
    answered = [r for r in answerable if r["grounded"]]
    trig = [r for r in rows if r["needs_research"]]
    ctrl = [r for r in rows if not r["needs_research"]]
    summary = {
        "n_tasks": len(rows),
        "retrieval_recall_at_10": _mean([r["recall_at_10"] for r in answerable]),
        "retrieval_recall_at_5": _mean([r["recall_at_5"] for r in answerable]),
        "baseline_top1_page_hit": _mean([r["top1_page_hit"] for r in answerable]),
        "task_success_rate": _mean([r["ok"] for r in rows]),
        "verifiable_accuracy": _mean([r["ok"] for r in verifiable]),
        "abstention_accuracy": _mean([r["ok"] for r in rows if r["abstain_task"]]),
        "false_abstention_rate": _mean([not r["grounded"] for r in answerable]),
        "groundedness_mean": _mean([r["groundedness"] for r in answered]),
        "citation_accuracy_mean": _mean([r["citation_accuracy"] for r in answered]),
        "cited_gold_page_rate": _mean([r["cited_gold_page"] for r in answered]),
        "re_search_rate_trigger": _mean([r["researched"] for r in trig]),
        "re_search_rate_control": _mean([r["researched"] for r in ctrl]),
        "mean_latency_s": _mean([r["latency_s"] for r in rows]),
    }
    if trace_path.exists() and trace_path.stat().st_size:
        summary["trace_complete_rate"] = round(metrics.trace_completeness(trace_path)["rate"], 4)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    parser.add_argument("--tasks", default=str(ROOT / "grading_kit/tasks.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "reports/eval_results.json"))
    parser.add_argument("--trace", default=str(ROOT / "traces/run.jsonl"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="a.b=value")
    parser.add_argument("--ablate", action="store_true", help="also run the one-stage-off ablations")
    args = parser.parse_args()

    cfg = apply_overrides(config.load(args.config), args.overrides)
    tasks = load_tasks(Path(args.tasks))
    retriever = Retriever(cfg)
    result = evaluate(cfg, tasks, retriever, Path(args.trace))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    out = {"config_overrides": args.overrides, **result}
    if args.ablate:
        scratch = Path(tempfile.mkdtemp()) / "ablation_trace.jsonl"
        out["ablation"] = ablation.run(
            cfg, lambda c: evaluate(c, tasks, retriever, scratch, verbose=False)["summary"])
        logging_conf.TRACE_PATH = Path(args.trace)
        print(json.dumps(out["ablation"], ensure_ascii=False, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())

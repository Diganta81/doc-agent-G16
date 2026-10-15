"""Kaggle check for the A3 stages that need the real index + BGE-M3 (too heavy for a CPU laptop).

Run it in a Kaggle notebook (GPU on, internet on) and paste back kaggle_a3_results.json:

    !git clone https://github.com/nazmul-islam00/doc-agent-G16 /kaggle/working/repo     # or add as a dataset
    %cd /kaggle/working/repo
    !pip install -q sentence-transformers faiss-cpu pydantic pydantic-settings pyyaml
    # upload the folder holding index.faiss + chunks.jsonl + metadata.json as a Kaggle dataset, then:
    !python scripts/kaggle_a3_check.py --index /kaggle/input/<your-index-dataset> --device cuda

Optional (real LLM instead of the extractive baseline): add LLM_API_KEY as a Kaggle secret / env var and
    ... --set llm.provider=openai_compat --set llm.base_url=https://api.groq.com/openai/v1 --set llm.model=<model>

Every step is wrapped, so one failure does not hide the others; errors are recorded in the JSON.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_eval  # noqa: E402
from doc_agent import config  # noqa: E402
from doc_agent.eval import ablation, metrics  # noqa: E402
from doc_agent.retrieval import rerank as rerank_mod  # noqa: E402
from doc_agent.retrieval.retriever import Retriever  # noqa: E402

RESULTS: dict = {}


def step(name):
    """Run a function, store its return value (or the traceback) under RESULTS[name]."""
    def wrap(fn):
        print(f"\n=== {name} ===", flush=True)
        started = time.perf_counter()
        try:
            RESULTS[name] = fn()
        except Exception:                                   # noqa: BLE001 - report, never abort
            RESULTS[name] = {"error": traceback.format_exc()}
            print(RESULTS[name]["error"], flush=True)
        RESULTS[name + "_seconds"] = round(time.perf_counter() - started, 1) if isinstance(RESULTS[name], dict) else None
        return fn
    return wrap


def ranked_pages(chunks) -> list[str]:
    seen, out = set(), []
    for c in chunks:
        for p in c.page_ids:
            b = run_eval._base(p)
            if b not in seen:
                seen.add(b)
                out.append(b)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="dir with index.faiss, chunks.jsonl, metadata.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--rerankers", default="cross-encoder/ms-marco-MiniLM-L6-v2,BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out", default="kaggle_a3_results.json")
    args = ap.parse_args()

    cfg = config.load(ROOT / "configs/config.yaml")
    cfg["index"]["path"] = args.index
    cfg.setdefault("embed", {})["device"] = args.device
    cfg["retrieve"]["rerank_device"] = args.device
    run_eval.apply_overrides(cfg, args.overrides)
    tasks = run_eval.load_tasks(ROOT / "grading_kit/tasks.jsonl")
    answerable = [t for t in tasks if not t.get("abstain") and t.get("gold_pages")]
    retriever = Retriever(cfg)

    @step("0_environment")
    def _env():
        info = {"python": platform.python_version(), "overrides": args.overrides, "device_requested": args.device}
        try:
            import torch
            info.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        except ImportError:
            info["torch"] = None
        meta = json.loads((Path(args.index) / "metadata.json").read_text(encoding="utf-8"))
        chunks = retriever.all_chunks()
        info["index_metadata"] = meta
        info["n_chunks"] = len(chunks)
        info["n_pages_in_index"] = len({run_eval._base(p) for c in chunks for p in c.page_ids})
        info["sample_page_ids"] = chunks[0].page_ids if chunks else None
        info["gold_pages_present"] = {
            p: p in {run_eval._base(q) for c in chunks for q in c.page_ids}
            for p in sorted({g for t in answerable for g in t["gold_pages"]})}
        return info

    @step("1_dense_retrieval_and_threshold")
    def _dense():
        """First-pass top score per task: calibrates retrieve.weak_threshold, and recall@k of gold pages."""
        rows, ks = [], (1, 5, 10, 20, 40)
        for t in tasks:
            chunks = retriever.retrieve(t["question"], 40)
            pages = ranked_pages(chunks)
            gold = [run_eval._base(p) for p in t.get("gold_pages", [])]
            top_at_10 = max((c.score for c in chunks[:10]), default=0.0)
            rows.append({"id": t["id"], "needs_research": t["needs_research"], "abstain": t.get("abstain", False),
                         "top_score@10": round(top_at_10, 4), "top_score@40": round(max(c.score for c in chunks), 4),
                         "top_pages": pages[:5], "gold": gold,
                         **{f"recall@{k}": metrics.recall_at_k(pages, gold, k) for k in ks if gold}})
        ans = [r for r in rows if not r["abstain"] and not r["needs_research"]]
        trig = [r for r in rows if r["needs_research"] and not r["abstain"]]
        neg = [r for r in rows if r["abstain"]]
        summary = {
            "weak_threshold_in_config": cfg["retrieve"]["weak_threshold"],
            "top_score_control_min": min((r["top_score@10"] for r in ans), default=None),
            "top_score_trigger_max": max((r["top_score@10"] for r in trig), default=None),
            "top_score_abstain_max": max((r["top_score@10"] for r in neg), default=None),
            "mean_recall": {f"recall@{k}": round(sum(r.get(f"recall@{k}", 0) for r in rows if r["gold"]) /
                                                max(1, sum(1 for r in rows if r["gold"])), 4) for k in ks},
        }
        lo, hi = summary["top_score_abstain_max"], summary["top_score_control_min"]
        if lo is not None and hi is not None:
            summary["threshold_hint"] = (round((lo + hi) / 2, 3) if lo < hi
                                         else "overlap: abstain scores reach control scores - threshold alone cannot separate")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return {"summary": summary, "rows": rows}

    @step("2_reranker_comparison")
    def _rerank():
        """Does a cross-encoder improve ranking of the gold page among the dense top-20?"""
        out = {}
        pool = {t["id"]: retriever.retrieve(t["question"], 20) for t in answerable}
        gold = {t["id"]: [run_eval._base(p) for p in t["gold_pages"]] for t in answerable}

        def score(order_fn):
            r1 = r5 = 0.0
            for t in answerable:
                pages = ranked_pages(order_fn(t))
                r1 += metrics.recall_at_k(pages, gold[t["id"]], 1)
                r5 += metrics.recall_at_k(pages, gold[t["id"]], 5)
            n = max(1, len(answerable))
            return {"recall@1": round(r1 / n, 4), "recall@5": round(r5 / n, 4)}

        out["dense_only"] = score(lambda t: pool[t["id"]])
        for name in [n for n in args.rerankers.split(",") if n]:
            c = copy.deepcopy(cfg)
            c["retrieve"].update(rerank=True, reranker=name)
            started = time.perf_counter()
            try:
                out[name] = score(lambda t, c=c: rerank_mod.rerank(t["question"], pool[t["id"]], c))
                out[name]["seconds_for_all_tasks"] = round(time.perf_counter() - started, 1)
            except Exception:                               # noqa: BLE001
                out[name] = {"error": traceback.format_exc()}
        print(json.dumps(out, indent=2))
        return out

    @step("3_agent_run_with_trace")
    def _agent():
        trace = ROOT / "traces" / "run.jsonl"
        res = run_eval.evaluate(cfg, tasks, retriever, trace)
        print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
        gate = {
            "trigger_k_paths": {r["id"]: r["k_path"] for r in res["rows"] if r["needs_research"]},
            "control_k_paths": {r["id"]: r["k_path"] for r in res["rows"] if not r["needs_research"]},
            "wrongly_researched_controls": [r["id"] for r in res["rows"] if not r["needs_research"] and r["researched"]],
            "triggers_that_did_not_research": [r["id"] for r in res["rows"] if r["needs_research"] and not r["researched"]],
            "trace_completeness": metrics.trace_completeness(trace),
            "trace_file": str(trace),
        }
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return {"summary": res["summary"], "gate": gate, "rows": res["rows"]}

    @step("4_ablation")
    def _ablation():
        scratch = Path(tempfile.mkdtemp()) / "abl.jsonl"
        out = ablation.run(cfg, lambda c: run_eval.evaluate(c, tasks, retriever, scratch, verbose=False)["summary"])
        print(json.dumps(out["delta"], indent=2))
        return out

    Path(args.out).write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  (also copy traces/run.jsonl back)")


if __name__ == "__main__":
    main()

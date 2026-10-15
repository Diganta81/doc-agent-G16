"""Stage 9 — ablation harness"""
from __future__ import annotations

import copy
from typing import Callable

from ..contracts import *  # noqa

# name -> dotted config path and the value that switches that stage OFF.
VARIANTS: dict[str, tuple[str, object]] = {
    "no_rerank": ("retrieve.rerank", False),
    "no_research": ("agent.research", False),      # single retrieval pass, no widening
    "no_reformulation": ("agent.reformulate", False),
}


def _with(cfg: dict, dotted: str, value) -> dict:
    out = copy.deepcopy(cfg)
    node = out
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value
    return out


def run(cfg: dict, evaluate: Callable[[dict], dict] | None = None) -> dict:
    """Toggle each stage off; report metric deltas.

    ``evaluate(cfg) -> {metric: number}`` runs the agent over the task suite (scripts/run_eval.py
    supplies it). Result: {"baseline": {...}, "<variant>": {...}, "delta": {"<variant>": {...}}}.
    """
    if evaluate is None:
        raise ValueError("ablation.run needs an evaluate(cfg) callable (see scripts/run_eval.py)")
    results = {"baseline": evaluate(cfg)}
    deltas: dict[str, dict] = {}
    for name, (path, off_value) in VARIANTS.items():
        results[name] = evaluate(_with(cfg, path, off_value))
        deltas[name] = {
            m: round(results[name][m] - results["baseline"][m], 4)
            for m in results["baseline"]
            if isinstance(results["baseline"][m], (int, float)) and m in results[name]
        }
    results["delta"] = deltas
    return results

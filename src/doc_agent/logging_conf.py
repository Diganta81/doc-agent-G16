"""FIXED — structured logging (auditable NFR). Use get_logger(), never print()."""
from __future__ import annotations
import json, logging, os, sys, time, uuid
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('{"ts":"%(asctime)s","lvl":"%(levelname)s","mod":"%(name)s","msg":"%(message)s"}'))
        lg.addHandler(h); lg.setLevel(logging.INFO)
    return lg


TRACE_PATH = Path(os.environ.get("DOC_AGENT_TRACE", "traces/run.jsonl"))
_BULKY = {"chunks", "text", "query"}      # kept out of obs: large, or already recorded in args


def _args_summary(args: dict) -> dict:
    out = {}
    for key, value in args.items():
        if key == "candidates":
            out[key] = [c["id"] if isinstance(c, dict) else str(c) for c in value]
        elif isinstance(value, str) and len(value) > 500:
            out[key] = value[:500] + "..."
        else:
            out[key] = value
    return out


def _obs_summary(result) -> dict:
    """What the agent observed from one tool call (decide() reads top_score + k from this)."""
    if result is None:
        return {}
    if not result.ok:
        return {"ok": False, **result.payload}
    return {k: v for k, v in result.payload.items() if k not in _BULKY}


def register(hooks) -> None:
    """Wire structured tracing at each seam (auditable trail) AND emit traces/run.jsonl.

    One TraceStep-shaped line (plus run_id) per tool call, written when its observation is known
    (the next ON_STEP, or BEFORE/AFTER_ANSWER), then one final "answer" line per run. Records pass
    through the ON_LOG seam so PII redaction applies to the audit trail as well.
    """
    cur: dict = {"state": None, "pending": None}

    def _write(state: dict, tool: str, args: dict, obs: dict) -> None:
        state["trace_step"] = state.get("trace_step", 0) + 1
        record = {"run_id": state["run_id"], "step": state["trace_step"], "tool": tool,
                  "args": args, "obs": obs, "ts": round(time.time(), 3)}
        record = hooks.run(hooks.ON_LOG, {"record": record}).get("record", record)
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _flush(obs: dict) -> None:
        pending = cur["pending"]
        if pending is not None and cur["state"] is not None:
            _write(cur["state"], pending["tool"], pending["args"], obs)
        cur["pending"] = None

    def _on_step(ctx: dict) -> dict:
        state = ctx["state"]
        if "run_id" not in state:
            state["run_id"] = uuid.uuid4().hex[:8]
        cur["state"] = state
        if cur["pending"] is not None and state.get("obs"):
            _flush(_obs_summary(state["obs"][-1]))
        return ctx

    def _on_tool_call(ctx: dict) -> dict:
        action = ctx["action"]
        cur["pending"] = {"tool": action["tool"], "args": _args_summary(action.get("args", {}))}
        return ctx

    def _on_before_answer(ctx: dict) -> dict:
        state = ctx["state"]
        cur["state"] = state
        if cur["pending"] is not None and state.get("obs"):
            _flush(_obs_summary(state["obs"][-1]))
        return ctx

    def _on_after_answer(ctx: dict) -> dict:
        state = cur["state"]
        if state is None:                       # blocked before the first step ran
            state = {"run_id": uuid.uuid4().hex[:8]}
            cur["state"] = state
        if ctx.get("blocked"):
            _flush({"blocked": ctx["blocked"]})
        answer = ctx["answer"]
        obs = {"abstained": not answer.grounded, "grounded": answer.grounded,
               "n_citations": len(answer.citations), "confidence": answer.confidence}
        if state.get("stop_reason"):
            obs["stop_reason"] = state["stop_reason"]
        if ctx.get("blocked"):
            obs["blocked"] = ctx["blocked"]
        _write(state, "answer", {}, obs)
        return ctx

    hooks.register(hooks.ON_STEP, _on_step)
    hooks.register(hooks.ON_TOOL_CALL, _on_tool_call)
    hooks.register(hooks.BEFORE_ANSWER, _on_before_answer)
    hooks.register(hooks.AFTER_ANSWER, _on_after_answer)

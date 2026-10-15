"""Stage 6 — SECURITY — autonomy, budgets, prompt-injection defense"""
from __future__ import annotations

import re

from ..contracts import *  # noqa


class GuardrailViolation(RuntimeError):
    """An action was blocked (budget, autonomy, or injection). ``kind`` says which."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


READ_ONLY_TOOLS = {"retrieve", "rerank", "read_page", "extract", "aggregate", "cite",
                   "calculator", "escalate_to_human"}
STATE_CHANGING_TOOLS = {"enhance_page"}   # rewrites derived page images -> needs approval

# Instruction-like text. Matched on retrieved passages and on tool arguments.
_INJECTION = re.compile(
    r"ignore (all |any |your |the )?(previous |prior |above )?(instructions?|rules?|prompts?)"
    r"|disregard (all |any |your |the )?(previous |prior |above )?(instructions?|rules?)"
    r"|you are now\b|system prompt|reveal (your|the) (prompt|instructions|key)"
    r"|new instructions?:|override (the )?(rules|instructions)"
    r"|পূর্বের (সব )?নির্দেশ(না)?(গুলি|াবলি)? (উপেক্ষা|বাদ|ভুলে)"
    r"|(আগের|উপরের) (সব )?নির্দেশ(না)?(গুলি|াবলি)? (উপেক্ষা|অগ্রাহ্য|ভুলে)",
    re.IGNORECASE,
)


def looks_like_injection(text: str) -> bool:
    return bool(text) and _INJECTION.search(text) is not None


def sanitize_text(text: str) -> tuple[str, bool]:
    """Mask instruction-like spans in untrusted document text. Returns (text, was_flagged).

    The mask has the same length as the original span so citation character offsets stay valid.
    """
    cleaned, n = _INJECTION.subn(lambda m: "▒" * len(m.group()), text or "")
    return cleaned, n > 0


class Guardrails:
    """Enforce autonomy level, step/cost budget, and instruction/content isolation."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["agent"]
        self.reset()

    def reset(self) -> None:
        self.spent = 0.0
        self.steps = 0

    def check(self, action: dict) -> None:
        """Raise GuardrailViolation if over budget / disallowed autonomy / injection detected."""
        tool = action.get("tool")
        known = READ_ONLY_TOOLS | STATE_CHANGING_TOOLS
        if tool not in known:
            raise GuardrailViolation("unknown_tool", f"tool {tool!r} is not in the tool registry")

        self.steps += 1
        max_steps = int(self.cfg.get("max_steps", 8))
        if self.steps > max_steps:
            raise GuardrailViolation("step_budget", f"more than {max_steps} tool calls")

        self.spent += float(self.cfg.get("cost_per_call_usd", 0.0))
        budget = float(self.cfg.get("budget_usd", 0.05))
        if self.spent > budget:
            raise GuardrailViolation("cost_budget", f"spent {self.spent:.4f} USD > budget {budget}")

        autonomy = str(self.cfg.get("autonomy", "act-with-approval"))
        if tool in STATE_CHANGING_TOOLS and autonomy != "autonomous" and not action.get("approved"):
            raise GuardrailViolation("approval_required", f"{tool} needs human approval")

        for value in _strings(action.get("args", {})):
            if looks_like_injection(value):
                raise GuardrailViolation("injection", "instruction-like text in tool arguments")


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def register(hooks, cfg: dict) -> None:
    """Wire guardrails into every tool call (ON_TOOL_CALL) and retrieved text (AFTER_RETRIEVE)."""
    g = Guardrails(cfg)

    def _check(ctx: dict) -> dict:
        g.check(ctx["action"])                    # budgets / autonomy / injection
        return ctx

    def _isolate(ctx: dict) -> dict:
        """Retrieved passages are DATA: strip instruction-like text before any model sees them."""
        flagged = 0
        cleaned = []
        for chunk in ctx.get("chunks", []):
            text, hit = sanitize_text(chunk.text)
            flagged += hit
            cleaned.append(chunk.model_copy(update={"text": text}) if hit else chunk)
        ctx["chunks"] = cleaned
        ctx["injection_flagged"] = flagged
        return ctx

    hooks.register(hooks.ON_TOOL_CALL, _check)
    hooks.register(hooks.AFTER_RETRIEVE, _isolate)

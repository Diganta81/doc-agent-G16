"""Stage 6 - FIXED loop - perceive -> decide -> act -> observe, with cross-cutting seams.
Implement decide() and synthesize() only. Security, grounding, PII, and tracing run via hooks at the
marked seams - do NOT inline them here."""
from __future__ import annotations
from ..contracts import *  # noqa
from .. import hooks
from ..governance import pii
from ..llm import postprocess, prompts
from ..llm.client import LLM, LLMUnavailable
from ..retrieval import retriever as retriever_mod
from . import hitl
from .guardrails import GuardrailViolation
from .memory import Memory
from .tools import ToolContext, build_tools

class Agent:
    """FIXED loop. Implement decide() (the policy) and synthesize() only."""
    def __init__(self, cfg: dict, retriever, llm: LLM | None = None, tools: dict | None = None) -> None:
        self.full_cfg = cfg
        self.cfg = cfg["agent"]; self.retriever = retriever; self.mem = Memory()
        self.llm = llm or LLM(cfg)
        self.tools = tools if tools is not None else build_tools(ToolContext(cfg, retriever))

    def run(self, query_text: str) -> Answer:
        state = {"query": query_text, "obs": []}
        for _ in range(self.cfg["max_steps"]):
            hooks.run(hooks.ON_STEP, {"state": state})
            action = self.decide(state)                      # IMPLEMENT (policy)
            if action["tool"] == "stop":
                break
            hooks.run(hooks.ON_TOOL_CALL, {"action": action})   # guardrails/injection/trace
            result = self.act(action)                        # runs the tool via REGISTRY
            state["obs"].append(result); self.mem.add(result)
        hooks.run(hooks.BEFORE_ANSWER, {"state": state})     # grounding gate / PII redact
        ans = self.synthesize(state)                         # IMPLEMENT (grounded answer)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})       # trace / metrics
        return ans

    def answer_safely(self, query_text: str) -> Answer:
        """run(), but a blocked action (budget / approval / injection) becomes an abstention.

        An approval-required block is queued for a human reviewer. The audit trail still gets its
        AFTER_ANSWER record so a blocked run is never an unlogged run.
        """
        try:
            return self.run(query_text)
        except GuardrailViolation as violation:
            if violation.kind == "approval_required":
                hitl.escalate(str(violation), {"query": query_text})
            ans = postprocess.abstain()
            hooks.run(hooks.AFTER_ANSWER, {"answer": ans, "blocked": violation.kind})
            return ans

    def decide(self, state: dict) -> dict:
        """Evidence-gated re-search — the MANDATORY agentic behaviour (A3 gate, fail-closed).
        Read the last observation (top_score, k) and branch on the NUMBER, using retrieval.retriever:
          1. retrieve at k = cfg.retrieve.k
          2. if is_weak(chunks, cfg):  k2 = next_k(k, cfg)
               - k2 is not None -> retrieve AGAIN at the wider k2 (widen the net), then re-check
               - k2 is None (hit k_max) and still weak -> ABSTAIN ("insufficient evidence")
          3. else -> synthesize a grounded, cited answer
        Emit obs {"top_score": ..., "k": ...} on each step. A fixed retrieve->answer path is NOT agentic
        and caps the grade. Rule-based (baseline) or RL policy (Stage 7)."""
        cfg = self.full_cfg
        query = state["query"]
        obs = state["obs"]
        if not obs:
            return {"tool": "retrieve", "args": {"query": query, "k": int(cfg["retrieve"]["k"])}}

        last = obs[-1]
        if not last.ok:
            state["stop_reason"] = "tool_error"
            return {"tool": "stop", "args": {}}
        payload = last.payload
        if "reranked" in payload:                      # rerank already ran -> evidence is final
            state["stop_reason"] = "reranked"
            return {"tool": "stop", "args": {}}

        chunks = [Chunk.model_validate(c) for c in payload["chunks"]]
        if retriever_mod.is_weak(chunks, cfg):
            k2 = retriever_mod.next_k(int(payload["k"]), cfg)
            if k2 is None or not self.cfg.get("research", True):
                state["stop_reason"] = "abstain_weak_evidence"      # k_max reached, still weak
                return {"tool": "stop", "args": {}}
            attempt = sum(1 for o in obs if "reranked" not in o.payload)
            return {"tool": "retrieve",
                    "args": {"query": self._reformulate(query, attempt), "k": k2}}

        if cfg["retrieve"].get("rerank"):
            return {"tool": "rerank", "args": {"query": query, "candidates": payload["chunks"]}}
        state["stop_reason"] = "evidence_ok"
        return {"tool": "stop", "args": {}}

    def _reformulate(self, query: str, attempt: int) -> str:
        """Rewrite the query for the wider search; fall back to the original on any LLM problem."""
        if not self.cfg.get("reformulate", True) or not self.llm.available:
            return query
        try:
            rewritten = self.llm.complete(
                prompts.DECIDE.format(question=query, attempt=attempt), max_tokens=64)
        except LLMUnavailable:
            return query
        rewritten = rewritten.strip().splitlines()[0].strip() if rewritten.strip() else ""
        return rewritten or query

    def act(self, action: dict) -> ToolResult:
        tool = self.tools.get(action["tool"])
        if tool is None:
            return ToolResult(ok=False, payload={"error": f"unknown tool {action['tool']!r}"})
        return tool(**action.get("args", {}))

    def synthesize(self, state: dict) -> Answer:
        """Grounded, cited answer; abstain if unsupported (no-hallucination)."""
        cfg = self.full_cfg
        if not postprocess.evidence_sufficient(state, cfg):
            return postprocess.abstain()

        evidence = postprocess.evidence_chunks(state)[: int(self.cfg.get("evidence_chunks", 5))]
        question = state["query"]
        raw = None
        if self.llm.available:
            block = "\n\n".join(
                f"[C{i}] (page {', '.join(c.page_ids)})\n{c.text}" for i, c in enumerate(evidence, 1))
            try:
                raw = self.llm.complete(prompts.SYNTHESIZE.format(question=question, evidence=block))
            except LLMUnavailable:
                raw = None
        if raw is None:                                   # no LLM: extractive baseline answer
            raw = self._extractive(question, evidence)

        text, citations = postprocess.parse_citations(raw, evidence)
        answer = postprocess.format_answer(
            text, citations, evidence=evidence,
            top_score=max(c.score for c in evidence),
            min_support=float(self.cfg.get("min_support", 0.3)))
        return answer.model_copy(update={"text": pii.redact(answer.text)})

    @staticmethod
    def _extractive(question: str, evidence: list[Chunk]) -> str:
        """Best-matching passage sentence, cited. Baseline when no LLM is configured."""
        wanted = set(postprocess.tokens(question))
        best, best_score = None, -1
        for index, chunk in enumerate(evidence, 1):
            for match in postprocess._SENTENCE.finditer(chunk.text):
                sentence = match.group().strip()
                if not sentence:
                    continue
                score = len(wanted & set(postprocess.tokens(sentence)))
                if score > best_score:
                    best, best_score = (sentence, index), score
        if best is None:
            return postprocess.INSUFFICIENT
        return f"{best[0]} [C{best[1]}]"

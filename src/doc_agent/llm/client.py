"""LLM — the single LLM call wrapper (all model calls go through here)"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable

from ..contracts import *  # noqa
from ..settings import settings


class LLMUnavailable(RuntimeError):
    """No usable LLM (provider 'none', missing key, or the call failed)."""


class LLM:
    """Model set by cfg['llm'] (OpenAI-compatible chat endpoint). Key comes from settings.

    provider "none" disables generation: callers fall back to extractive behaviour. ``transport``
    can be injected (tests) and replaces the HTTP call: transport(url, headers, body) -> dict.
    """

    def __init__(self, cfg: dict, transport: Callable[[str, dict, dict], dict] | None = None) -> None:
        self.cfg = cfg
        self.llm_cfg = cfg.get("llm", {})
        self.provider = str(self.llm_cfg.get("provider", "none")).lower()
        self._injected = transport is not None
        self._transport = transport or self._http
        self.calls = 0

    @property
    def available(self) -> bool:
        return self.provider != "none" and bool(settings.llm_api_key or self._injected)

    def complete(self, prompt: str, **kw) -> str:
        if not self.available:
            raise LLMUnavailable("LLM provider is 'none' or no API key is configured")
        body = {
            "model": self.llm_cfg.get("model", ""),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kw.get("temperature", self.llm_cfg.get("temperature", 0.0)),
            "max_tokens": kw.get("max_tokens", self.llm_cfg.get("max_tokens", 512)),
        }
        url = str(self.llm_cfg.get("base_url", "")).rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.llm_api_key}",
        }
        self.calls += 1
        try:
            data = self._transport(url, headers, body)
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(f"unexpected LLM response shape: {exc}") from exc

    def _http(self, url: str, headers: dict, body: dict) -> dict:
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        timeout = float(self.llm_cfg.get("timeout_s", 60))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(f"LLM request failed: {exc}") from exc

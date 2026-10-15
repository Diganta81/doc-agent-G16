"""LLM — the single LLM call wrapper (all model calls go through here)"""
from __future__ import annotations

import json
import time
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
        self.failures = 0
        self.last_error = ""

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
        except LLMUnavailable as exc:
            self._record_failure(exc)
            raise
        except (KeyError, IndexError, TypeError) as exc:
            error = LLMUnavailable(f"unexpected LLM response shape: {exc}")
            self._record_failure(error)
            raise error from exc

    def _record_failure(self, exc: Exception) -> None:
        """Count failures so an eval run can show how many answers silently fell back to extractive."""
        self.failures += 1
        self.last_error = str(exc)[:300]

    def _http(self, url: str, headers: dict, body: dict) -> dict:
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        timeout = float(self.llm_cfg.get("timeout_s", 60))
        retries = int(self.llm_cfg.get("retries", 3))
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(2 ** attempt * 2)           # rate limit / transient: back off, retry
                    continue
                detail = exc.read().decode("utf-8", "replace")[:200]
                raise LLMUnavailable(f"LLM request failed: HTTP {exc.code} {detail}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise LLMUnavailable(f"LLM request failed: {exc}") from exc
        raise LLMUnavailable("LLM request failed after retries")

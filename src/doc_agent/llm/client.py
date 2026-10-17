"""LLM — the single LLM call wrapper (all model calls go through here)"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
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
        self.cache_hits = 0
        self._cache: dict[str, str] | None = None
        self._last_call = 0.0
        self.last_error = ""

    @property
    def available(self) -> bool:
        return self.provider != "none" and bool(settings.llm_api_key.strip() or self._injected)

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
            "Authorization": f"Bearer {settings.llm_api_key.strip()}",
        }
        key = self._cache_key(body)
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self._throttle()
        self.calls += 1
        try:
            data = self._transport(url, headers, body)
            text = str(data["choices"][0]["message"]["content"]).strip()
            self._cache_put(key, text)
            return text
        except LLMUnavailable as exc:
            self._record_failure(exc)
            raise
        except (KeyError, IndexError, TypeError) as exc:
            error = LLMUnavailable(f"unexpected LLM response shape: {exc}")
            self._record_failure(error)
            raise error from exc

    def _cache_key(self, body: dict) -> str:
        payload = json.dumps([body["model"], body["messages"], body["temperature"], body["max_tokens"]],
                             ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_file(self) -> Path | None:
        path = self.llm_cfg.get("cache_path")
        return Path(path) if path else None

    def _cache_get(self, key: str) -> str | None:
        """Replies are cached on disk (llm.cache_path): reruns and ablations skip calls already paid for."""
        file = self._cache_file()
        if file is None:
            return None
        if self._cache is None:
            self._cache = {}
            if file.exists():
                for line in file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        self._cache[row["k"]] = row["v"]
        return self._cache.get(key)

    def _cache_put(self, key: str, text: str) -> None:
        file = self._cache_file()
        if file is None or not text:                     # never cache failures or empty replies
            return
        self._cache_get(key)                             # make sure the file was loaded first
        self._cache[key] = text
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"k": key, "v": text}, ensure_ascii=False) + "\n")

    def _throttle(self) -> None:
        """Keep at least llm.min_interval_s between calls (free-tier requests-per-minute limits)."""
        wait = float(self.llm_cfg.get("min_interval_s", 0)) - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

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
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                raise LLMUnavailable(f"LLM request failed: {exc}") from exc
        raise LLMUnavailable("LLM request failed after retries")

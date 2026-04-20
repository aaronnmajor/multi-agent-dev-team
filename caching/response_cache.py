"""
Simple in-memory response cache keyed by a SHA-256 hash of the full prompt.

Useful when the same tool schema or system prompt is sent repeatedly and the
LLM response is deterministic for that input. For more advanced caching
(provider-level prefix caching, semantic similarity) see the Week 4 course
notes — those are left as future work.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from observability.logging import get_logger

_log = get_logger("cache")


class ResponseCache:
    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[str, float]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(system: str, messages: list[dict[str, Any]]) -> str:
        payload = json.dumps({"system": system, "messages": messages}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, system: str, messages: list[dict[str, Any]]) -> str | None:
        key = self._key(system, messages)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        value, stored_at = entry
        if time.monotonic() - stored_at >= self.ttl_seconds:
            self._store.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        _log.info("cache_hit", key_prefix=key[:8])
        return value

    def set(self, system: str, messages: list[dict[str, Any]], value: str) -> None:
        self._store[self._key(system, messages)] = (value, time.monotonic())

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

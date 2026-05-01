"""
Two-tier memory system for the Coder Agent.

- SlidingWindowBuffer: FIFO short-term buffer of recent conversation turns.
- SemanticMemory: ChromaDB-backed long-term store with embedding-based retrieval.

Embeddings and LLM calls are routed through Helicone for usage tracking.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

SLIDING_WINDOW_SIZE = 20
TOP_K_MEMORIES = 3

_HELICONE_BASE = os.getenv("HELICONE_BASE_URL")
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")

helicone_client = OpenAI(
    api_key=_OPENROUTER_API_KEY,
    base_url=_HELICONE_BASE,
    default_headers={"Helicone-Auth": f"Bearer {_HELICONE_API_KEY}"},
)


class HeliconeEmbeddingFunction(EmbeddingFunction):
    """Embedding function that routes through the Helicone proxy.

    Implements the small required surface for Chroma's EmbeddingFunction
    contract (``__init__``, ``__call__``, ``name``) so the deprecation
    warnings don't fire on newer Chroma versions.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model

    @staticmethod
    def name() -> str:
        return "helicone"

    def __call__(self, input: Documents) -> Embeddings:
        response = helicone_client.embeddings.create(
            model=self.model,
            input=input,
        )
        return [item.embedding for item in response.data]


class SlidingWindowBuffer:
    """Keeps the N most recent conversation turns; the system message is preserved separately."""

    def __init__(self, max_turns: int = SLIDING_WINDOW_SIZE) -> None:
        self._max_turns = max_turns
        self._system: dict[str, Any] | None = None
        self._turns: deque[dict[str, Any]] = deque()

    def set_system(self, content: str) -> None:
        self._system = {"role": "system", "content": content}

    def add(self, role: str, content: str) -> None:
        self._turns.append({"role": role, "content": content})
        while len(self._turns) > self._max_turns:
            self._turns.popleft()

    def messages(self) -> list[dict[str, Any]]:
        base = [self._system] if self._system else []
        return base + list(self._turns)


class SemanticMemory:
    """ChromaDB-backed long-term store with cosine-similarity retrieval.

    When ``CHROMA_HOST`` is set in the environment (e.g., from docker-compose
    where a chroma service is running), connects via the HTTP client so memory
    persists across pipeline runs. Otherwise falls back to an in-process
    Client, which keeps unit tests fast and avoids a network dependency.
    """

    def __init__(self, collection_name: str = "coder_agent_memory") -> None:
        chroma_host = os.getenv("CHROMA_HOST")
        if chroma_host:
            chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
            self._client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=HeliconeEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        self._counter = 0

    def store(self, text: str, metadata: dict | None = None) -> str:
        self._counter += 1
        mem_id = f"mem_{self._counter:04d}"
        self._collection.add(
            ids=[mem_id],
            documents=[text],
            metadatas=[metadata or {"source": "agent"}],
        )
        return mem_id

    def retrieve(self, query: str, top_k: int = TOP_K_MEMORIES) -> list[str]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
        )
        return results.get("documents", [[]])[0]

    def count(self) -> int:
        return self._collection.count()

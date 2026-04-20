"""
Agent-to-Agent (A2A) protocol — in-process implementation.

Provides the five-field Message schema, a validating Broker with asyncio queues,
and capability advertisement helpers. This demonstrates the A2A contract
(Coder ↔ QA peer messaging) without the HTTP-based service discovery a
production deployment would require.

The LangGraph graph in `orchestration/graph.py` uses direct shared-state routing
for simplicity. This module exists as a parallel primitive that could be swapped
in for async peer communication in future weeks.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

VALID_AGENTS = frozenset({"coder", "qa", "pm"})
VALID_INTENTS = frozenset({"review_request", "fix_instruction", "approved"})


@dataclass
class Message:
    """A2A message with the five mandatory fields per the protocol spec."""

    sender:         str
    receiver:       str
    intent:         str
    payload:        dict
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if self.sender not in VALID_AGENTS:
            raise ValueError(f"Unknown sender: {self.sender!r}. Valid: {sorted(VALID_AGENTS)}")
        if self.receiver not in VALID_AGENTS:
            raise ValueError(f"Unknown receiver: {self.receiver!r}. Valid: {sorted(VALID_AGENTS)}")
        if self.intent not in VALID_INTENTS:
            raise ValueError(f"Unknown intent: {self.intent!r}. Valid: {sorted(VALID_INTENTS)}")
        if self.sender == self.receiver:
            raise ValueError("sender and receiver must differ")


class Broker:
    """Per-agent asyncio queues; senders enqueue by receiver name, receivers pull from their own queue."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Message]] = {
            agent: asyncio.Queue() for agent in VALID_AGENTS
        }

    async def send(self, message: Message) -> None:
        await self._queues[message.receiver].put(message)

    async def receive(self, agent_name: str) -> Message:
        if agent_name not in self._queues:
            raise ValueError(f"Unknown agent: {agent_name!r}")
        return await self._queues[agent_name].get()

    def pending(self, agent_name: str) -> int:
        """How many messages are queued for this agent (for observability)."""
        return self._queues[agent_name].qsize() if agent_name in self._queues else 0


# Capability advertisements — each agent declares the intents it supports on startup.
AGENT_CAPABILITIES: dict[str, dict[str, list[str]]] = {
    "coder": {
        "sends":    ["review_request"],
        "receives": ["fix_instruction", "approved"],
    },
    "qa": {
        "sends":    ["fix_instruction", "approved"],
        "receives": ["review_request"],
    },
    "pm": {
        "sends":    [],
        "receives": ["fix_instruction"],  # Escalation path for cross-task inconsistencies.
    },
}


def validate_incoming(message: Message, receiver: str, allowed_senders: set[str]) -> bool:
    """Trust-boundary check: receiver validates sender + intent before processing.

    Returns False for unknown intents or unauthorised senders so the caller can
    log and discard rather than raise (which would crash the agent).
    """
    if message.receiver != receiver:
        return False
    if message.sender not in allowed_senders:
        return False
    supported = AGENT_CAPABILITIES.get(receiver, {}).get("receives", [])
    return message.intent in supported


__all__ = [
    "Message",
    "Broker",
    "AGENT_CAPABILITIES",
    "VALID_AGENTS",
    "VALID_INTENTS",
    "validate_incoming",
]

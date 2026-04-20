"""
MCP (Model Context Protocol) adapter.

This module exposes the project's tools through an MCP-style interface:
  - A registry of tool descriptors (name, description, input schema)
  - A single dispatch function that invokes a tool by name with validated arguments

It's not a full MCP server (no stdio transport, no JSON-RPC framing), but it
demonstrates the same contract — a uniform way for any LLM agent (ours, or one
on the other end of an MCP connection) to discover and invoke tools.

In Week 4 this module is the natural place to hook up a real MCP server or
A2A protocol bridge; the tool functions themselves don't have to change.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from tools.code_executor import exec_python
from tools.file_io import read_file, write_file


class MCPToolDescriptor(TypedDict):
    """A single tool advertised over the MCP interface."""

    name: str
    description: str
    input_schema: dict[str, Any]


# The LangChain `@tool`-decorated functions expose their schema via `.args_schema`
# and their callable via `.invoke`. We translate those into MCP descriptors here.

def _descriptor(fn: Any) -> MCPToolDescriptor:
    schema = fn.args_schema.model_json_schema() if getattr(fn, "args_schema", None) else {}
    return {
        "name":         fn.name,
        "description":  fn.description or "",
        "input_schema": schema,
    }


TOOL_REGISTRY: dict[str, Any] = {
    "read_file":   read_file,
    "write_file":  write_file,
    "exec_python": exec_python,
}


def list_tools() -> list[MCPToolDescriptor]:
    """Return MCP-style descriptors for every registered tool."""
    return [_descriptor(fn) for fn in TOOL_REGISTRY.values()]


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Invoke a tool by name with the given arguments. Always returns a string.

    Equivalent to the `tools/call` method on a real MCP server.
    """
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'. Available: {list(TOOL_REGISTRY.keys())}"
    try:
        return fn.invoke(arguments)
    except Exception as e:
        return f"Error invoking tool '{name}': {type(e).__name__}: {e}"


__all__ = ["MCPToolDescriptor", "TOOL_REGISTRY", "list_tools", "call_tool"]

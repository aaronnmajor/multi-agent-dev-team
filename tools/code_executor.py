"""Sandboxed Python code executor using subprocess with configurable timeout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool

MAX_OUTPUT_CHARS = 2000
MAX_TIMEOUT = 30
DEFAULT_TIMEOUT = 10

WORKSPACE = Path("workspace")


@tool
def exec_python(path: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Execute a Python script from the workspace in a sandboxed subprocess.

    Args:
        path: Relative path to the .py file (inside the workspace).
        timeout: Maximum seconds to allow (clamped to 30). Defaults to 10.
    """
    try:
        effective_timeout = min(timeout, MAX_TIMEOUT)
        target = WORKSPACE / path
        result = subprocess.run(
            [sys.executable, str(target)],
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )
        if result.returncode == 0:
            output = result.stdout
            if len(output) > MAX_OUTPUT_CHARS:
                output = output[:MAX_OUTPUT_CHARS] + "\n[output truncated]"
            return output if output else "(no output -- script ran successfully)"
        stderr = result.stderr
        if len(stderr) > MAX_OUTPUT_CHARS:
            stderr = stderr[:MAX_OUTPUT_CHARS] + "\n[output truncated]"
        return f"Error (exit code {result.returncode}):\n{stderr}"
    except subprocess.TimeoutExpired:
        return f"Error: script '{path}' exceeded timeout of {effective_timeout}s and was killed."
    except FileNotFoundError:
        return f"Error: script '{path}' not found in workspace."
    except Exception as e:
        return f"Error executing '{path}': {type(e).__name__}: {e}"

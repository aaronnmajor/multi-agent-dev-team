"""File I/O tools: read and write files, confined to the workspace directory."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool

WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)


@tool
def read_file(path: str) -> str:
    """Read the contents of a file from the workspace.

    Args:
        path: Relative path to the file (inside the workspace).
    """
    try:
        target = WORKSPACE / path
        with open(target, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error reading file '{path}': No such file or directory."
    except PermissionError:
        return f"Error reading file '{path}': Permission denied."
    except Exception as e:
        return f"Error reading file '{path}': {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace, creating parent directories as needed.

    Args:
        path: Relative path to the target file (inside the workspace).
        content: Full text content to write.
    """
    try:
        target = WORKSPACE / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        line_count = len(content.splitlines())
        return f"File written successfully. {line_count} lines."
    except PermissionError:
        return f"Error writing file '{path}': Permission denied."
    except Exception as e:
        return f"Error writing file '{path}': {e}"

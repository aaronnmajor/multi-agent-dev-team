"""
Post-pipeline verification of agent-generated artifacts.

After the LangGraph pipeline finishes, the Coder has written one or more
files into ``workspace/`` and the QA agent has reviewed them. The QA
review is correctness-by-LLM-judgement; this module adds a stronger
correctness signal by actually running pytest against any test files
the agent produced.

The result is structured (counts + raw output) so it can be embedded in
the cost report or printed at the end of a CLI run.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKSPACE_DEFAULT = Path("workspace")
PYTEST_TIMEOUT_S = 60


def _has_test_files(workspace: Path) -> bool:
    """True when at least one ``test_*.py`` or ``*_test.py`` exists under workspace."""
    for pattern in ("test_*.py", "*_test.py"):
        if any(workspace.rglob(pattern)):
            return True
    return False


def _parse_pytest_summary(stdout: str) -> dict[str, int]:
    """Extract pass/fail/error counts from pytest's terminal summary line."""
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    # Match the standard "N passed, M failed, K errors in T.TTs" line.
    match = re.search(
        r"(\d+) passed|(\d+) failed|(\d+) errors|(\d+) skipped",
        stdout,
    )
    if not match:
        return counts
    for entry in re.finditer(r"(\d+)\s+(passed|failed|errors?|skipped)", stdout):
        n, label = int(entry.group(1)), entry.group(2)
        key = "errors" if label.startswith("error") else label
        counts[key] = n
    return counts


def verify_workspace(workspace: Path | str = WORKSPACE_DEFAULT) -> dict[str, Any]:
    """Run pytest against the workspace and return a structured result.

    Returns a dict with:
      - ``status``: ``"pass"`` (all tests passed) | ``"fail"`` (≥1 failure) |
                    ``"no_tests"`` | ``"error"`` (pytest itself errored / timeout)
      - ``passed``, ``failed``, ``errors``, ``skipped``: counts (zero on no_tests)
      - ``stdout``: trimmed pytest output (last 2000 chars)
    """
    workspace = Path(workspace)
    if not workspace.exists():
        return {"status": "no_tests", "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "stdout": ""}
    if not _has_test_files(workspace):
        return {"status": "no_tests", "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "stdout": ""}

    cmd = [
        sys.executable, "-m", "pytest",
        str(workspace),
        "-q",
        "--no-header",
        "-rN",
        # The project's root conftest excludes workspace/ from collection so
        # the main test suite stays clean. We bypass that exclusion here so
        # the agent-generated tests actually run.
        "--noconftest",
        f"--rootdir={workspace.resolve()}",
        f"--timeout={PYTEST_TIMEOUT_S}",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PYTEST_TIMEOUT_S * 2,
        )
    except subprocess.TimeoutExpired:
        return {
            "status":  "error",
            "passed":  0,
            "failed":  0,
            "errors":  0,
            "skipped": 0,
            "stdout":  "pytest invocation timed out",
        }

    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    counts = _parse_pytest_summary(out)
    if proc.returncode == 0 and counts["passed"] >= 0 and counts["failed"] == 0 and counts["errors"] == 0:
        status = "pass"
    elif proc.returncode == 5:
        # pytest exit 5 = "no tests collected"
        status = "no_tests"
    else:
        status = "fail"
    return {
        "status":  status,
        "passed":  counts["passed"],
        "failed":  counts["failed"],
        "errors":  counts["errors"],
        "skipped": counts["skipped"],
        "stdout":  out[-2000:],
    }


def format_verification(result: dict[str, Any]) -> str:
    """Render a one-line summary of a verify_workspace result for CLI output."""
    status = result["status"]
    if status == "no_tests":
        return "VERIFY: no test files in workspace/ — skipping pytest."
    if status == "error":
        return f"VERIFY: pytest errored — {result.get('stdout', '')[-200:]}"
    return (
        f"VERIFY: {status.upper()} — "
        f"passed={result['passed']} failed={result['failed']} "
        f"errors={result['errors']} skipped={result['skipped']}"
    )

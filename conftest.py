"""pytest configuration for the multi-agent-dev-team project."""

import pytest_asyncio  # noqa: F401 — imported for its side effect of registering the asyncio plugin

# Agent-generated artifacts land in workspace/ during e2e runs; those files are
# not project tests and should never be collected. norecursedirs is honoured by
# pytest's collector regardless of whether the user passes a path.
collect_ignore_glob = ["workspace/**", "docs/api/**"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm: marks tests that make real LLM API calls (slow, costs tokens)"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks tests that use asyncio (handled by pytest-asyncio)"
    )

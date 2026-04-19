"""pytest configuration for the multi-agent-dev-team project."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm: marks tests that make real LLM API calls (slow, costs tokens)"
    )

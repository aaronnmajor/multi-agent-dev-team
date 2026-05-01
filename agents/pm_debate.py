"""
Debate / Consensus layer for the Product Manager phase.

When ``PM_DEBATE_MODE`` is set in the environment, the PM agent runs three
LLM calls instead of one before producing the tech spec:

1. An *advocate-for-simplicity* PM proposes the leanest spec it can.
2. An *advocate-for-completeness* PM proposes a spec that prioritises
   covering every edge case, even at the cost of extra tasks.
3. A *synthesiser* reads both proposals and produces a single tech spec
   that takes the strongest argument from each, justified inline.

The synthesiser's spec is what flows into ``decompose_into_tasks``. The
debate adds two extra LLM calls per pipeline run, which is cheap on the
PM-only path and well worth it for ambiguous requirements where a single
PM pass tends to either over-spec or under-spec.

Disabled by default; opt in via ``PM_DEBATE_MODE=true``.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from observability import get_logger, record_usage_from_response, trace_span

_log = get_logger("pm_debate")


def is_enabled() -> bool:
    """Return True when the debate flow should be used."""
    return os.getenv("PM_DEBATE_MODE", "false").lower() == "true"


SIMPLICITY_PROMPT = """You are a Product Manager whose guiding principle is RADICAL SIMPLICITY.

Convert the user's raw requirement into the leanest possible Markdown technical specification.
Your bias is: cut anything that isn't strictly necessary, prefer the smallest viable surface
area, and choose conventions that minimise lines of code. If two designs work, pick the one
with fewer files and fewer abstractions.

Produce a Markdown document with exactly these five sections:

# Overview
# Functional Requirements
# Non-Functional Requirements
# File Structure
# Constraints and Assumptions

End with a one-paragraph "Why simplicity wins here" justifying your choices.
"""


COMPLETENESS_PROMPT = """You are a Product Manager whose guiding principle is RIGOROUS COMPLETENESS.

Convert the user's raw requirement into a Markdown technical specification that addresses
every reasonable edge case, error path, and validation concern up front. Your bias is: a
small amount of extra effort up front prevents large amounts of rework later. Surface
ambiguities explicitly and resolve them with conservative defaults.

Produce a Markdown document with exactly these five sections:

# Overview
# Functional Requirements
# Non-Functional Requirements
# File Structure
# Constraints and Assumptions

End with a one-paragraph "Why completeness wins here" justifying your choices.
"""


SYNTHESIS_PROMPT = """You are a senior Product Manager arbitrating between two competing draft specs.

You will receive:
  - The user's original requirement.
  - Draft A, written by a PM whose bias is RADICAL SIMPLICITY.
  - Draft B, written by a PM whose bias is RIGOROUS COMPLETENESS.

Produce ONE final tech spec that takes the strongest argument from each draft. Where the
two drafts disagree, choose the position that better serves the user's stated requirement
and explain the call in a short inline note (italicised). Where they agree, fold the
agreement into the spec without commentary.

Output a Markdown document with exactly these five sections — no preamble, no fences:

# Overview
# Functional Requirements
# Non-Functional Requirements
# File Structure
# Constraints and Assumptions

After the five sections, append a "## Synthesis notes" subsection that lists the 2-4 most
important calls you made and why.
"""


def _propose(
    role_prompt: str,
    requirement: str,
    model: str,
    client: OpenAI,
    run_id: str,
    span_name: str,
) -> str:
    """Run a single debate-role LLM call and return its raw content."""
    with trace_span("pm_debate", span_name, run_id):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": f"Requirement:\n\n{requirement}"},
            ],
        )
    record_usage_from_response(run_id, "pm", model, response)
    return (response.choices[0].message.content or "").strip()


def propose_spec_with_debate(
    requirement: str,
    model: str,
    client: OpenAI,
    run_id: str = "",
) -> dict[str, Any]:
    """Run the three-call debate flow and return the synthesised spec.

    Returns a dict with ``spec`` (the final markdown), ``draft_a`` (simplicity
    proposal), and ``draft_b`` (completeness proposal) so callers can log or
    inspect what each side argued for. Empty drafts fall back to a single
    PM pass on the surviving draft so a partial debate never blocks the run.
    """
    draft_a = _propose(
        SIMPLICITY_PROMPT, requirement, model, client, run_id, "advocate_simplicity",
    )
    draft_b = _propose(
        COMPLETENESS_PROMPT, requirement, model, client, run_id, "advocate_completeness",
    )

    if not draft_a and not draft_b:
        return {"spec": "", "draft_a": draft_a, "draft_b": draft_b}
    if not draft_a:
        return {"spec": draft_b, "draft_a": draft_a, "draft_b": draft_b}
    if not draft_b:
        return {"spec": draft_a, "draft_a": draft_a, "draft_b": draft_b}

    synthesis_input = (
        f"Original requirement:\n{requirement}\n\n"
        f"--- Draft A (simplicity advocate) ---\n{draft_a}\n\n"
        f"--- Draft B (completeness advocate) ---\n{draft_b}\n"
    )
    with trace_span("pm_debate", "synthesise_spec", run_id):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYNTHESIS_PROMPT},
                {"role": "user", "content": synthesis_input},
            ],
        )
    record_usage_from_response(run_id, "pm", model, response)
    spec = (response.choices[0].message.content or "").strip()

    _log.info(
        "pm_debate_complete",
        run_id=run_id,
        draft_a_chars=len(draft_a),
        draft_b_chars=len(draft_b),
        spec_chars=len(spec),
    )
    return {"spec": spec, "draft_a": draft_a, "draft_b": draft_b}

# Reflection — Multi-Agent Dev Team

> Capstone post-mortem for *Build Autonomous Multi-Agent Systems*
> (Saras AI Institute, Spring 2026). What worked, what broke, what I'd do
> differently next time.

## Three design decisions worth examining

### 1. Reducer-typed lists for `ProjectState`

The shared state declares three list fields with `Annotated[list, add]`
reducers — `tasks`, `artifacts`, and `reviews`. The default LangGraph
behaviour for a non-annotated list field is "overwrite on every update",
which seems convenient until you hit the QA-then-Coder review loop in
Week 3. The first time the QA agent routed back to the Coder, the Coder
ran the same task again and emitted a fresh `artifacts` list containing
only that artifact — the old artifacts disappeared. The reducer fix is one
import and four characters per field, but recognising the failure mode
required reading the LangGraph source. The lesson is that LangGraph's
explicitness is a feature: the framework forces you to declare your
intent at the type level, and once you do, append-only semantics become
a load-bearing invariant the rest of the system can rely on.

### 2. Tiered models with env-var overrides

`config.AGENT_MODELS` defaults to `gpt-4o` for the PM and `gpt-4o-mini`
for the Coder and QA. The per-agent split is justified by what each
agent actually does: the PM converts an underspecified requirement into
an authoritative spec — a single ambiguity at this stage cascades into
every downstream task. The Coder and QA, by contrast, work on already-
structured inputs and produce verifiable outputs, so a cheaper mid-tier
model is sufficient. Overlaying env-var overrides on top of the tiered
defaults turned out to be the right shape for cost-optimisation
experiments: I could rerun the same demo with `MODEL_PM=gpt-4o-mini` to
measure the floor, then compare against the default tier to size the
quality gain. Keeping the model strings entirely outside agent logic
also paid off when I added Innovation #2's debate layer — three calls,
zero changes to the call sites.

### 3. Self-reflection as a single pass rather than an iterative loop

The Reflexion paper proposes an iterative critique loop where the
agent rewrites its output until a verifier accepts it. I deliberately
chose a single-pass critique instead. Two reasons. First, the QA agent
is already an external verifier with a structured rubric and a 2-retry
budget — adding another internal loop would duplicate that mechanism
and double the LLM cost without a clear quality justification. Second,
single-pass critique is bounded: the worst-case latency is one extra
LLM call per task, predictable, and easy to disable via env var when I
want clean token measurements. The trade-off is that the Coder won't
catch deep correctness issues that need multiple iterations to fix —
but the QA agent's retry loop already covers that case from a different
angle. Layering two iteration mechanisms felt like a recipe for
non-determinism.

## The hardest bug

The hardest bug wasn't a crash; it was a silent correctness failure in
the QA agent's parser. The QA reviewer returns JSON and the parser
defaults to `passed=True` on a `JSONDecodeError`. That choice is
deliberate — a transient parse failure should not deadlock the
pipeline — but it created a confusing failure mode during Week 3
development: a syntactically broken QA response would let through a
genuinely broken artifact, and the only sign was a slightly odd
`summary` field saying the response was malformed. I caught it by
reading the logs, not by running tests, because the tests were happy
to pass on a default-pass review.

The fix wasn't to change the default. The fix was to log the malformed
case at WARN level with the raw payload, and to add a unit test that
asserts the default-pass behaviour and the summary-string contract. The
takeaway: graceful degradation is correct, but it has to be observable.
A silent "open the door anyway" is worse than a noisy crash.

## What I'd do differently

**Tests-first from Day One.** The Week 4 test suite is solid — 79
deterministic mocked tests as of the production-ready milestone — but
those tests were written *after* each agent was implemented. If I'd
started the project with the test scaffolding in place, the QA-parser
bug above would have surfaced immediately rather than in Week 4 when I
finally wrote a coverage matrix and noticed the gap.

**Model diversity earlier.** Every LLM call in this project goes
through Helicone to OpenRouter to OpenAI. The course allows Claude as
well, and the QA agent in particular would benefit from a different
model family — running PM and QA on the same model risks shared
blind spots in spec interpretation. Adding Anthropic alongside OpenAI
would have been a half-day of plumbing if I'd done it in Week 2;
retrofitting it post-Week 4 was no longer worth the effort against
the rubric.

**Smaller iteration ceilings to start.** `MAX_ITERATIONS = 10` for the
Coder was an overestimate. In practice the Coder converges in 3-5
iterations on every demo task, and the extra ceiling just inflated
worst-case token spend during runaway debugging. A leaner default with
an explicit override would have been more honest about the actual
operating envelope.

---

*Word count: ~770 words. Author: Aaron Major. Date: 2026-05-01.*

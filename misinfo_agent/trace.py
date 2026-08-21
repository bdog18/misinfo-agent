"""Phase 3: Investigation / TraceStep / Evidence dataclasses.

Plain dataclasses (not pydantic) — unlike eval/schema.py and tools.py,
these aren't validating untrusted external input, they're just structured
records our own code builds up as an investigation runs, and they need to
serialize cleanly to JSON (via dataclasses.asdict) for the run logs under
runs/. Keep every field JSON-primitive (str/int/float/list/dict/None) —
no datetimes, no nested pydantic models.
"""

from dataclasses import dataclass, field
from typing import Literal

from misinfo_agent.eval.schema import Verdict

Stance = Literal["supports", "refutes", "irrelevant", "mixed"]


@dataclass
class TraceStep:
    """One turn of the ReAct loop: what the model thought, did, and observed.

    input_tokens/output_tokens are the token usage for the model call that
    produced this step's thought/action/action_input (the orchestrator's
    reasoning call) — this is what lets the eval harness compare the
    agent arm's cost against the baseline's.
    """

    step_number: int
    thought: str
    action: str
    action_input: dict
    observation: str
    input_tokens: int
    output_tokens: int


@dataclass
class Evidence:
    """One source the agent weighed against the claim, distilled from a
    tool_compare_claim_to_text + tool_assess_source call pair.

    This is what the eval harness scores directly for "did the agent seek
    disconfirming evidence" and "were its sources credible" — kept separate
    from raw TraceSteps so score.py doesn't need to know how to parse tool
    call args back out of the full trace.
    """

    url: str
    stance: Stance
    confidence: float
    quote: str | None
    source_credibility: str
    source_bias: str
    step_number: int


@dataclass
class Investigation:
    """A full run of one arm (agent or baseline) against one claim."""

    claim: str
    arm: Literal["agent", "baseline"]
    model: str
    steps: list[TraceStep] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    verdict: Verdict | None = None
    confidence: float | None = None
    reasoning: str | None = None
    stop_reason: Literal["verdict_submitted", "max_steps_reached"] | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def add_step(
        self,
        *,
        thought: str,
        action: str,
        action_input: dict,
        observation: str,
        input_tokens: int,
        output_tokens: int,
    ) -> TraceStep:
        """Build a TraceStep, auto-numbering it, append it to self.steps, and return it."""
        trace = TraceStep(
            step_number=len(self.steps) + 1,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.steps.append(trace)
        return trace  

    def add_evidence(self, evidence: Evidence) -> None:
        """Record one distilled piece of evidence."""
        self.evidence.append(evidence)

    @property
    def total_input_tokens(self) -> int:
        """Sum of input_tokens across every step so far."""
        return sum(step.input_tokens for step in self.steps)

    @property
    def total_output_tokens(self) -> int:
        """Sum of output_tokens across every step so far."""
        return sum(step.output_tokens for step in self.steps)

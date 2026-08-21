import dataclasses
import json

from misinfo_agent.trace import Evidence, Investigation, TraceStep


def _investigation(**overrides) -> Investigation:
    base = dict(claim="The measles vaccine causes autism.", arm="agent", model="claude-sonnet-5")
    base.update(overrides)
    return Investigation(**base)


def test_new_investigation_starts_empty():
    inv = _investigation()
    assert inv.steps == []
    assert inv.evidence == []
    assert inv.verdict is None
    assert inv.total_input_tokens == 0
    assert inv.total_output_tokens == 0


def test_add_step_numbers_from_one():
    inv = _investigation()

    step = inv.add_step(
        thought="I should search for this claim.",
        action="tool_search",
        action_input={"query": "measles vaccine autism"},
        observation="[]",
        input_tokens=100,
        output_tokens=20,
    )

    assert isinstance(step, TraceStep)
    assert step.step_number == 1
    assert inv.steps == [step]


def test_add_step_increments_step_number_across_calls():
    inv = _investigation()

    inv.add_step(
        thought="t1", action="tool_search", action_input={}, observation="o1",
        input_tokens=10, output_tokens=5,
    )
    second = inv.add_step(
        thought="t2", action="tool_fetch", action_input={}, observation="o2",
        input_tokens=15, output_tokens=8,
    )

    assert second.step_number == 2
    assert [s.step_number for s in inv.steps] == [1, 2]


def test_total_tokens_sum_across_steps():
    inv = _investigation()
    inv.add_step(
        thought="t1", action="tool_search", action_input={}, observation="o1",
        input_tokens=100, output_tokens=20,
    )
    inv.add_step(
        thought="t2", action="tool_fetch", action_input={}, observation="o2",
        input_tokens=50, output_tokens=10,
    )

    assert inv.total_input_tokens == 150
    assert inv.total_output_tokens == 30


def test_add_evidence_appends():
    inv = _investigation()
    ev = Evidence(
        url="https://www.cdc.gov/measles/about.html",
        stance="refutes",
        confidence=0.9,
        quote="There is no evidence linking the measles vaccine to autism.",
        source_credibility="high",
        source_bias="center",
        step_number=2,
    )

    inv.add_evidence(ev)

    assert inv.evidence == [ev]


def test_investigation_is_json_serializable_via_asdict():
    inv = _investigation()
    inv.add_step(
        thought="t1", action="tool_search", action_input={"query": "x"}, observation="o1",
        input_tokens=10, output_tokens=5,
    )
    inv.add_evidence(
        Evidence(
            url="https://example.com",
            stance="supports",
            confidence=0.5,
            quote=None,
            source_credibility="unknown",
            source_bias="unknown",
            step_number=1,
        )
    )
    inv.verdict = "false"
    inv.confidence = 0.8
    inv.reasoning = "The primary source directly contradicts the claim."

    # Should not raise — every field must be a JSON-primitive type.
    json.dumps(dataclasses.asdict(inv))

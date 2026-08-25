import dataclasses
import json
from unittest.mock import MagicMock, patch

from misinfo_agent import tools
from misinfo_agent.agent import ORCHESTRATOR_MODEL
from misinfo_agent.baseline import BASELINE_SYSTEM_PROMPT, run_baseline
from misinfo_agent.trace import Investigation

FAKE_SEARCH_RESULTS = [
    tools.SearchResult(
        title="CDC on measles",
        url="https://www.cdc.gov/measles",
        snippet="No link between the measles vaccine and autism has been found.",
        score=0.9,
    ),
]


def _fake_verdict_response(verdict="false", confidence=0.95, reasoning="No credible evidence supports the claim."):
    """A forced tool_choice response: response.content[0] IS the tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.input = {"verdict": verdict, "confidence": confidence, "reasoning": reasoning}
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock(input_tokens=500, output_tokens=50)
    return response


@patch("misinfo_agent.tools._get_anthropic_client")
@patch("misinfo_agent.tools.tool_search")
def test_run_baseline_returns_investigation_with_correct_arm_and_model(mock_search, mock_get_client):
    mock_search.return_value = FAKE_SEARCH_RESULTS
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_verdict_response()
    mock_get_client.return_value = mock_client

    investigation = run_baseline("The measles vaccine causes autism.")

    assert isinstance(investigation, Investigation)
    assert investigation.claim == "The measles vaccine causes autism."
    assert investigation.arm == "baseline"
    assert investigation.model == ORCHESTRATOR_MODEL


@patch("misinfo_agent.tools._get_anthropic_client")
@patch("misinfo_agent.tools.tool_search")
def test_run_baseline_extracts_verdict_confidence_and_reasoning(mock_search, mock_get_client):
    mock_search.return_value = FAKE_SEARCH_RESULTS
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_verdict_response(
        verdict="false", confidence=0.87, reasoning="Extensive research finds no link."
    )
    mock_get_client.return_value = mock_client

    investigation = run_baseline("The measles vaccine causes autism.")

    assert investigation.verdict == "false"
    assert investigation.confidence == 0.87
    assert investigation.reasoning == "Extensive research finds no link."
    assert investigation.stop_reason == "verdict_submitted"


@patch("misinfo_agent.tools._get_anthropic_client")
@patch("misinfo_agent.tools.tool_search")
def test_run_baseline_calls_tool_search_once_with_the_claim(mock_search, mock_get_client):
    mock_search.return_value = []
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_verdict_response()
    mock_get_client.return_value = mock_client

    run_baseline("some claim")

    mock_search.assert_called_once_with("some claim")


@patch("misinfo_agent.tools._get_anthropic_client")
@patch("misinfo_agent.tools.tool_search")
def test_run_baseline_forces_submit_verdict_tool_choice(mock_search, mock_get_client):
    mock_search.return_value = []
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_verdict_response()
    mock_get_client.return_value = mock_client

    run_baseline("some claim")

    _, kwargs = mock_client.messages.create.call_args
    assert kwargs["model"] == ORCHESTRATOR_MODEL
    assert kwargs["system"] == BASELINE_SYSTEM_PROMPT
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_verdict"}
    assert [t["name"] for t in kwargs["tools"]] == ["submit_verdict"]


@patch("misinfo_agent.tools._get_anthropic_client")
@patch("misinfo_agent.tools.tool_search")
def test_run_baseline_logs_exactly_two_steps_with_correct_token_accounting(mock_search, mock_get_client):
    mock_search.return_value = FAKE_SEARCH_RESULTS
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_verdict_response()
    mock_get_client.return_value = mock_client

    investigation = run_baseline("some claim")

    assert len(investigation.steps) == 2

    search_step, verdict_step = investigation.steps
    assert search_step.action == "tool_search"
    assert search_step.action_input == {"query": "some claim"}
    assert search_step.input_tokens == 0
    assert search_step.output_tokens == 0

    assert verdict_step.action == "submit_verdict"
    assert verdict_step.action_input == {
        "verdict": "false",
        "confidence": 0.95,
        "reasoning": "No credible evidence supports the claim.",
    }
    assert verdict_step.input_tokens == 500
    assert verdict_step.output_tokens == 50


@patch("misinfo_agent.tools._get_anthropic_client")
@patch("misinfo_agent.tools.tool_search")
def test_run_baseline_investigation_is_json_serializable(mock_search, mock_get_client):
    mock_search.return_value = FAKE_SEARCH_RESULTS
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_verdict_response()
    mock_get_client.return_value = mock_client

    investigation = run_baseline("some claim")

    # Every field (including step.observation, which holds the raw
    # SearchResult list) must have been stringified — a raw pydantic model
    # or list of them would break dataclasses.asdict()'s JSON round-trip.
    json.dumps(dataclasses.asdict(investigation))

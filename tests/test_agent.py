from unittest.mock import MagicMock, patch

from misinfo_agent import tools
from misinfo_agent.agent import (
    AGENT_TOOLS,
    MAX_HITS_PER_DOMAIN,
    ORCHESTRATOR_MODEL,
    SYSTEM_PROMPT,
    TOOL_DISPATCH,
    _domain_hit_count,
    _sought_disconfirming_evidence,
    run_investigation,
)
from misinfo_agent.trace import Evidence, Investigation

# ---------------------------------------------------------------------------
# AGENT_TOOLS — the schema surface the orchestrator sees
# ---------------------------------------------------------------------------

EXPECTED_TOOL_NAMES = {
    "tool_search",
    "tool_fetch",
    "tool_assess_source",
    "tool_compare_claim_to_text",
    "submit_verdict",
}


def _tool_by_name(name: str) -> dict:
    matches = [t for t in AGENT_TOOLS if t["name"] == name]
    assert len(matches) == 1, f"expected exactly one tool named {name!r}"
    return matches[0]


def test_agent_tools_has_exactly_the_expected_names():
    assert {t["name"] for t in AGENT_TOOLS} == EXPECTED_TOOL_NAMES


def test_every_tool_has_description_and_object_input_schema():
    for t in AGENT_TOOLS:
        assert t.get("description"), f"{t['name']} has no description"
        schema = t["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema


def test_tool_search_schema_has_query_required_and_max_results_optional():
    schema = _tool_by_name("tool_search")["input_schema"]
    assert "query" in schema["properties"]
    assert schema["properties"]["query"]["type"] == "string"
    assert "query" in schema["required"]
    assert "max_results" in schema["properties"]
    assert "max_results" not in schema.get("required", [])


def test_tool_fetch_schema_requires_url():
    schema = _tool_by_name("tool_fetch")["input_schema"]
    assert schema["properties"]["url"]["type"] == "string"
    assert schema["required"] == ["url"]


def test_tool_assess_source_schema_requires_url():
    schema = _tool_by_name("tool_assess_source")["input_schema"]
    assert schema["properties"]["url"]["type"] == "string"
    assert schema["required"] == ["url"]


def test_tool_compare_claim_to_text_schema_requires_claim_text_and_url():
    # "url" isn't a parameter of tools.tool_compare_claim_to_text itself — it's
    # bookkeeping-only, so the loop can attribute the resulting Evidence back
    # to the source it came from. Dispatch must strip it before calling the
    # real function.
    schema = _tool_by_name("tool_compare_claim_to_text")["input_schema"]
    assert schema["properties"]["claim"]["type"] == "string"
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["url"]["type"] == "string"
    assert set(schema["required"]) == {"claim", "text", "url"}


def test_submit_verdict_schema_requires_verdict_confidence_reasoning():
    schema = _tool_by_name("submit_verdict")["input_schema"]
    assert set(schema["required"]) == {"verdict", "confidence", "reasoning"}
    assert schema["properties"]["verdict"]["enum"] == ["true", "false", "mixed"]
    assert schema["properties"]["confidence"]["type"] == "number"
    assert schema["properties"]["reasoning"]["type"] == "string"


# ---------------------------------------------------------------------------
# TOOL_DISPATCH — mapping tool names to the real callables
# ---------------------------------------------------------------------------


def test_tool_dispatch_covers_the_four_real_tools_and_excludes_submit_verdict():
    assert set(TOOL_DISPATCH) == EXPECTED_TOOL_NAMES - {"submit_verdict"}


def test_tool_dispatch_maps_to_the_actual_tools_module_functions():
    assert TOOL_DISPATCH["tool_search"] is tools.tool_search
    assert TOOL_DISPATCH["tool_fetch"] is tools.tool_fetch
    assert TOOL_DISPATCH["tool_assess_source"] is tools.tool_assess_source
    assert TOOL_DISPATCH["tool_compare_claim_to_text"] is tools.tool_compare_claim_to_text


# ---------------------------------------------------------------------------
# System prompt + guardrails
# ---------------------------------------------------------------------------


def _investigation() -> Investigation:
    return Investigation(claim="The measles vaccine causes autism.", arm="agent", model="claude-sonnet-5")


def test_system_prompt_mentions_key_guardrail_concepts():
    assert "disconfirm" in SYSTEM_PROMPT.lower()
    assert str(MAX_HITS_PER_DOMAIN) in SYSTEM_PROMPT


def test_domain_hit_count_counts_only_tool_fetch_steps_on_that_domain():
    inv = _investigation()
    inv.add_step(
        thought="t", action="tool_fetch", action_input={"url": "https://www.cdc.gov/a"},
        observation="o", input_tokens=1, output_tokens=1,
    )
    inv.add_step(
        thought="t", action="tool_fetch", action_input={"url": "https://foxnews.com/b"},
        observation="o", input_tokens=1, output_tokens=1,
    )
    inv.add_step(
        thought="t", action="tool_fetch", action_input={"url": "https://cdc.gov/c"},
        observation="o", input_tokens=1, output_tokens=1,
    )
    # A non-fetch step mentioning "cdc.gov" in its query should not count.
    inv.add_step(
        thought="t", action="tool_search", action_input={"query": "site:cdc.gov measles"},
        observation="o", input_tokens=1, output_tokens=1,
    )

    assert _domain_hit_count(inv, "cdc.gov") == 2
    assert _domain_hit_count(inv, "foxnews.com") == 1
    assert _domain_hit_count(inv, "nytimes.com") == 0


def test_sought_disconfirming_evidence_false_when_no_evidence_yet():
    inv = _investigation()
    assert _sought_disconfirming_evidence(inv) is False


def test_sought_disconfirming_evidence_false_when_all_supports():
    inv = _investigation()
    inv.add_evidence(
        Evidence(
            url="https://example.com/a", stance="supports", confidence=0.9,
            quote=None, source_credibility="high", source_bias="center", step_number=1,
        )
    )
    inv.add_evidence(
        Evidence(
            url="https://example.com/b", stance="supports", confidence=0.7,
            quote=None, source_credibility="unknown", source_bias="unknown", step_number=2,
        )
    )
    assert _sought_disconfirming_evidence(inv) is False


def test_sought_disconfirming_evidence_true_when_a_refutes_entry_exists():
    inv = _investigation()
    inv.add_evidence(
        Evidence(
            url="https://example.com/a", stance="supports", confidence=0.9,
            quote=None, source_credibility="high", source_bias="center", step_number=1,
        )
    )
    inv.add_evidence(
        Evidence(
            url="https://example.com/b", stance="refutes", confidence=0.8,
            quote="Actually, this is false.", source_credibility="high",
            source_bias="center", step_number=2,
        )
    )
    assert _sought_disconfirming_evidence(inv) is True


def test_sought_disconfirming_evidence_true_for_mixed_stance():
    inv = _investigation()
    inv.add_evidence(
        Evidence(
            url="https://example.com/a", stance="mixed", confidence=0.6,
            quote=None, source_credibility="mixed", source_bias="center", step_number=1,
        )
    )
    assert _sought_disconfirming_evidence(inv) is True


# ---------------------------------------------------------------------------
# run_investigation — the loop mechanics
# ---------------------------------------------------------------------------


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(name: str, tool_input: dict, tool_use_id: str) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = tool_input
    block.id = tool_use_id
    return block


def _response(thought: str, tool_name: str, tool_input: dict, tool_use_id="toolu_1"):
    """One simulated Sonnet turn: a text block (the thought) + one tool_use block."""
    response = MagicMock()
    response.content = [_text_block(thought), _tool_use_block(tool_name, tool_input, tool_use_id)]
    response.usage = MagicMock(input_tokens=100, output_tokens=20)
    return response


def _patched_client(mock_get_client, side_effects):
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = side_effects
    mock_get_client.return_value = mock_client
    return mock_client


@patch("misinfo_agent.tools._get_anthropic_client")
def test_happy_path_ends_with_verdict_after_disconfirming_evidence(mock_get_client):
    responses = [
        _response("Let me search.", "tool_search", {"query": "measles vaccine autism"}, "t1"),
        _response("Let me fetch a result.", "tool_fetch", {"url": "https://www.cdc.gov/measles"}, "t2"),
        _response("Let me assess this source.", "tool_assess_source", {"url": "https://www.cdc.gov/measles"}, "t3"),
        _response(
            "Let me compare the claim to this text.",
            "tool_compare_claim_to_text",
            {"claim": "The measles vaccine causes autism.", "text": "No link found.", "url": "https://www.cdc.gov/measles"},
            "t4",
        ),
        _response(
            "I found disconfirming evidence, submitting.",
            "submit_verdict",
            {"verdict": "false", "confidence": 0.85, "reasoning": "CDC directly refutes the claim."},
            "t5",
        ),
    ]
    _patched_client(mock_get_client, responses)

    fake_search = MagicMock(return_value=[tools.SearchResult(
        title="CDC on measles", url="https://www.cdc.gov/measles", snippet="...", score=0.9,
    )])
    fake_fetch = MagicMock(return_value=tools.FetchResult(
        url="https://www.cdc.gov/measles", text="No link found.",
    ))
    fake_assess = MagicMock(return_value=tools.SourceAssessment(
        domain="cdc.gov", name="CDC", type="government", credibility="high", bias="center",
    ))
    fake_compare = MagicMock(return_value=tools.ComparisonResult(
        stance="refutes", confidence=0.9, quote="No link found.", reasoning="Direct refutation.",
    ))

    with patch.dict(
        "misinfo_agent.agent.TOOL_DISPATCH",
        {
            "tool_search": fake_search,
            "tool_fetch": fake_fetch,
            "tool_assess_source": fake_assess,
            "tool_compare_claim_to_text": fake_compare,
        },
    ):
        investigation = run_investigation("The measles vaccine causes autism.")

    assert investigation.verdict == "false"
    assert investigation.confidence == 0.85
    assert investigation.stop_reason == "verdict_submitted"
    assert len(investigation.steps) == 5

    # The bookkeeping "url" must be stripped before calling the real function.
    fake_compare.assert_called_once_with(
        claim="The measles vaccine causes autism.", text="No link found.",
    )

    assert len(investigation.evidence) == 1
    ev = investigation.evidence[0]
    assert ev.url == "https://www.cdc.gov/measles"
    assert ev.stance == "refutes"
    assert ev.source_credibility == "high"
    assert ev.source_bias == "center"
    assert ev.step_number == 4


@patch("misinfo_agent.tools._get_anthropic_client")
def test_max_steps_safety_cap_stops_the_loop(mock_get_client):
    # The model never calls submit_verdict — same tool_search call forever.
    responses = [
        _response("Searching again.", "tool_search", {"query": "measles vaccine autism"}, f"t{i}")
        for i in range(10)
    ]
    _patched_client(mock_get_client, responses)
    fake_search = MagicMock(return_value=[])

    with patch.dict("misinfo_agent.agent.TOOL_DISPATCH", {"tool_search": fake_search}):
        investigation = run_investigation("The measles vaccine causes autism.", max_steps=4)

    assert investigation.verdict is None
    assert investigation.stop_reason == "max_steps_reached"
    assert len(investigation.steps) == 4


@patch("misinfo_agent.tools._get_anthropic_client")
def test_submit_verdict_rejected_without_disconfirming_evidence_then_accepted(mock_get_client):
    responses = [
        _response(
            "Assessing the first source.", "tool_assess_source",
            {"url": "https://example.com/a"}, "t1",
        ),
        _response(
            "Comparing.", "tool_compare_claim_to_text",
            {"claim": "c", "text": "supports it", "url": "https://example.com/a"}, "t2",
        ),
        _response(
            "Submitting.", "submit_verdict",
            {"verdict": "true", "confidence": 0.9, "reasoning": "Looks true."}, "t3",
        ),
        _response(
            "Let me check another source.", "tool_assess_source",
            {"url": "https://example.com/b"}, "t4",
        ),
        _response(
            "Comparing the second source.", "tool_compare_claim_to_text",
            {"claim": "c", "text": "actually refutes it", "url": "https://example.com/b"}, "t5",
        ),
        _response(
            "Now submitting for real.", "submit_verdict",
            {"verdict": "mixed", "confidence": 0.6, "reasoning": "Mixed evidence found."}, "t6",
        ),
    ]
    _patched_client(mock_get_client, responses)

    fake_assess = MagicMock(side_effect=[
        tools.SourceAssessment(domain="example.com", type="unknown", credibility="unknown", bias="unknown"),
        tools.SourceAssessment(domain="example.com", type="unknown", credibility="unknown", bias="unknown"),
    ])
    fake_compare = MagicMock(side_effect=[
        tools.ComparisonResult(stance="supports", confidence=0.8, quote=None, reasoning="r1"),
        tools.ComparisonResult(stance="refutes", confidence=0.7, quote=None, reasoning="r2"),
    ])

    with patch.dict(
        "misinfo_agent.agent.TOOL_DISPATCH",
        {"tool_assess_source": fake_assess, "tool_compare_claim_to_text": fake_compare},
    ):
        investigation = run_investigation("some claim", max_steps=10)

    assert investigation.verdict == "mixed"
    assert investigation.stop_reason == "verdict_submitted"
    assert len(investigation.steps) == 6
    # The rejected attempt should be logged, but not as the thing that ended the loop.
    assert investigation.steps[2].action == "submit_verdict"
    assert "disconfirm" in investigation.steps[2].observation.lower()
    assert len(investigation.evidence) == 2


@patch("misinfo_agent.tools._get_anthropic_client")
def test_domain_hit_cap_blocks_dispatch_beyond_the_limit(mock_get_client):
    urls = [f"https://cdc.gov/page-{i}" for i in range(MAX_HITS_PER_DOMAIN + 1)]
    responses = [
        _response(f"Fetching page {i}.", "tool_fetch", {"url": url}, f"t{i}")
        for i, url in enumerate(urls)
    ]
    _patched_client(mock_get_client, responses)
    fake_fetch = MagicMock(
        side_effect=lambda url: tools.FetchResult(url=url, text="some text")
    )

    with patch.dict("misinfo_agent.agent.TOOL_DISPATCH", {"tool_fetch": fake_fetch}):
        investigation = run_investigation("some claim", max_steps=MAX_HITS_PER_DOMAIN + 1)

    assert investigation.stop_reason == "max_steps_reached"
    assert fake_fetch.call_count == MAX_HITS_PER_DOMAIN
    last_step = investigation.steps[-1]
    assert last_step.action == "tool_fetch"
    assert "cdc.gov" in last_step.observation.lower()


@patch("misinfo_agent.tools._get_anthropic_client")
def test_compare_rejected_without_prior_source_assessment(mock_get_client):
    responses = [
        _response(
            "Comparing without assessing first.", "tool_compare_claim_to_text",
            {"claim": "c", "text": "some text", "url": "https://example.com/unassessed"}, "t1",
        ),
    ]
    _patched_client(mock_get_client, responses)
    fake_compare = MagicMock()

    with patch.dict("misinfo_agent.agent.TOOL_DISPATCH", {"tool_compare_claim_to_text": fake_compare}):
        investigation = run_investigation("some claim", max_steps=1)

    fake_compare.assert_not_called()
    assert investigation.evidence == []
    assert "assess_source" in investigation.steps[0].observation.lower()
    assert "example.com/unassessed" in investigation.steps[0].observation


@patch("misinfo_agent.tools._get_anthropic_client")
def test_run_investigation_passes_exclude_domains_to_tool_search(mock_get_client):
    responses = [_response("Searching.", "tool_search", {"query": "some claim"}, "t1")]
    _patched_client(mock_get_client, responses)
    fake_search = MagicMock(return_value=[])

    with patch.dict("misinfo_agent.agent.TOOL_DISPATCH", {"tool_search": fake_search}):
        run_investigation("some claim", max_steps=1, exclude_domains={"politifact.com"})

    fake_search.assert_called_once_with(query="some claim", exclude_domains={"politifact.com"})


@patch("misinfo_agent.tools._get_anthropic_client")
def test_submit_verdict_coerces_non_string_verdict_to_lowercase_string(mock_get_client):
    # Real Anthropic tool calls have occasionally returned a JSON boolean
    # (False) for "verdict" instead of the string "false" - since the enum
    # values happen to collide with JSON's own boolean literals. Nothing
    # should end up storing that raw bool: investigation.verdict must
    # always be the string the rest of the codebase (and score.py) expects.
    responses = [
        _response("Assessing.", "tool_assess_source", {"url": "https://example.com/a"}, "t1"),
        _response(
            "Comparing.", "tool_compare_claim_to_text",
            {"claim": "c", "text": "refutes it", "url": "https://example.com/a"}, "t2",
        ),
        _response(
            "Submitting.", "submit_verdict",
            {"verdict": False, "confidence": 0.9, "reasoning": "Refuted."}, "t3",
        ),
    ]
    _patched_client(mock_get_client, responses)

    fake_assess = MagicMock(return_value=tools.SourceAssessment(
        domain="example.com", type="unknown", credibility="unknown", bias="unknown",
    ))
    fake_compare = MagicMock(return_value=tools.ComparisonResult(
        stance="refutes", confidence=0.9, quote=None, reasoning="r",
    ))

    with patch.dict(
        "misinfo_agent.agent.TOOL_DISPATCH",
        {"tool_assess_source": fake_assess, "tool_compare_claim_to_text": fake_compare},
    ):
        investigation = run_investigation("some claim", max_steps=5)

    assert investigation.verdict == "false"
    assert isinstance(investigation.verdict, str)


@patch("misinfo_agent.tools._get_anthropic_client")
def test_tool_fetch_rejected_for_excluded_domain(mock_get_client):
    responses = [
        _response(
            "Fetching the fact-check directly.", "tool_fetch",
            {"url": "https://politifact.com/factchecks/x"}, "t1",
        ),
    ]
    _patched_client(mock_get_client, responses)
    fake_fetch = MagicMock()

    with patch.dict("misinfo_agent.agent.TOOL_DISPATCH", {"tool_fetch": fake_fetch}):
        investigation = run_investigation(
            "some claim", max_steps=1, exclude_domains={"politifact.com"}
        )

    fake_fetch.assert_not_called()
    assert "REJECTED" in investigation.steps[0].observation
    assert "politifact.com" in investigation.steps[0].observation


@patch("misinfo_agent.tools._get_anthropic_client")
def test_tool_fetch_allowed_for_non_excluded_domain(mock_get_client):
    responses = [
        _response("Fetching.", "tool_fetch", {"url": "https://cdc.gov/page"}, "t1"),
    ]
    _patched_client(mock_get_client, responses)
    fake_fetch = MagicMock(return_value=tools.FetchResult(url="https://cdc.gov/page", text="ok"))

    with patch.dict("misinfo_agent.agent.TOOL_DISPATCH", {"tool_fetch": fake_fetch}):
        investigation = run_investigation(
            "some claim", max_steps=1, exclude_domains={"politifact.com"}
        )

    fake_fetch.assert_called_once_with(url="https://cdc.gov/page")
    assert "REJECTED" not in investigation.steps[0].observation


@patch("misinfo_agent.tools._get_anthropic_client")
def test_run_investigation_exclude_domains_defaults_to_none(mock_get_client):
    responses = [_response("Searching.", "tool_search", {"query": "some claim"}, "t1")]
    _patched_client(mock_get_client, responses)
    fake_search = MagicMock(return_value=[])

    with patch.dict("misinfo_agent.agent.TOOL_DISPATCH", {"tool_search": fake_search}):
        run_investigation("some claim", max_steps=1)

    fake_search.assert_called_once_with(query="some claim", exclude_domains=None)

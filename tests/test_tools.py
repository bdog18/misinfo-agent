from unittest.mock import MagicMock, patch

import httpx2
import pytest
from anthropic import APIConnectionError

from misinfo_agent.tools import (
    MAX_FETCH_CHARS,
    ComparisonResult,
    FetchResult,
    SearchResult,
    SourceAssessment,
    _extract_domain,
    fact_checker_domains,
    tool_assess_source,
    tool_compare_claim_to_text,
    tool_fetch,
    tool_search,
)

FAKE_TAVILY_RESPONSE = {
    "results": [
        {
            "title": "Fact-check: measles vaccine claim",
            "url": "https://www.cdc.gov/measles/about/questions.html",
            "content": "The measles vaccine does not cause autism...",
            "score": 0.91,
            "published_date": "2026-03-01",
        },
        {
            "title": "A blog post about vaccines",
            "url": "https://example-blog.com/post",
            "content": "Some opinion content...",
            "score": 0.42,
            # no published_date on this one — Tavily doesn't always include it
        },
    ]
}


@patch("misinfo_agent.tools._get_client")
def test_tool_search_returns_structured_results(mock_get_client):
    mock_client = MagicMock()
    mock_client.search.return_value = FAKE_TAVILY_RESPONSE
    mock_get_client.return_value = mock_client

    results = tool_search("measles vaccine autism claim")

    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].title == "Fact-check: measles vaccine claim"
    assert results[0].url == "https://www.cdc.gov/measles/about/questions.html"
    assert results[0].snippet.startswith("The measles vaccine")
    assert results[0].score == 0.91
    assert results[0].published_date == "2026-03-01"


@patch("misinfo_agent.tools._get_client")
def test_tool_search_missing_published_date_defaults_to_none(mock_get_client):
    mock_client = MagicMock()
    mock_client.search.return_value = FAKE_TAVILY_RESPONSE
    mock_get_client.return_value = mock_client

    results = tool_search("measles vaccine autism claim")

    assert results[1].published_date is None


@patch("misinfo_agent.tools._get_client")
def test_tool_search_passes_max_results_through(mock_get_client):
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}
    mock_get_client.return_value = mock_client

    tool_search("some query", max_results=3)

    mock_client.search.assert_called_once_with("some query", max_results=3)


@patch("misinfo_agent.tools._get_client")
def test_tool_search_empty_results_returns_empty_list(mock_get_client):
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}
    mock_get_client.return_value = mock_client

    assert tool_search("query with no hits") == []


@patch("misinfo_agent.tools._get_client")
def test_tool_search_excludes_matching_domains(mock_get_client):
    mock_client = MagicMock()
    mock_client.search.return_value = FAKE_TAVILY_RESPONSE
    mock_get_client.return_value = mock_client

    results = tool_search("measles vaccine autism claim", exclude_domains={"cdc.gov"})

    assert [r.url for r in results] == ["https://example-blog.com/post"]


@patch("misinfo_agent.tools._get_client")
def test_tool_search_exclude_domains_none_keeps_everything(mock_get_client):
    mock_client = MagicMock()
    mock_client.search.return_value = FAKE_TAVILY_RESPONSE
    mock_get_client.return_value = mock_client

    results = tool_search("measles vaccine autism claim", exclude_domains=None)

    assert len(results) == 2


def test_tool_search_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    import misinfo_agent.tools as tools_module

    tools_module._client = None  # reset the cached client so the missing-key path runs

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        tool_search("anything")


@patch("misinfo_agent.tools._get_client")
def test_tool_fetch_returns_text_on_success(mock_get_client):
    mock_client = MagicMock()
    mock_client.extract.return_value = {
        "results": [
            {
                "url": "https://www.cdc.gov/measles/about/questions.html",
                "raw_content": "The measles vaccine does not cause autism...",
            }
        ],
        "failed_results": [],
    }
    mock_get_client.return_value = mock_client

    result = tool_fetch("https://www.cdc.gov/measles/about/questions.html")

    assert isinstance(result, FetchResult)
    assert result.url == "https://www.cdc.gov/measles/about/questions.html"
    assert result.text == "The measles vaccine does not cause autism..."
    assert result.error is None


@patch("misinfo_agent.tools._get_client")
def test_tool_fetch_calls_extract_with_url_list(mock_get_client):
    mock_client = MagicMock()
    mock_client.extract.return_value = {"results": [], "failed_results": []}
    mock_get_client.return_value = mock_client

    tool_fetch("https://example.com/page")

    mock_client.extract.assert_called_once_with(urls=["https://example.com/page"])


@patch("misinfo_agent.tools._get_client")
def test_tool_fetch_returns_error_on_failed_url(mock_get_client):
    mock_client = MagicMock()
    mock_client.extract.return_value = {
        "results": [],
        "failed_results": [
            {"url": "https://example.com/gone", "error": "404: Not Found"},
        ],
    }
    mock_get_client.return_value = mock_client

    result = tool_fetch("https://example.com/gone")

    assert result.url == "https://example.com/gone"
    assert result.text is None
    assert result.error == "404: Not Found"


@patch("misinfo_agent.tools._get_client")
def test_tool_fetch_truncates_long_text(mock_get_client):
    mock_client = MagicMock()
    long_text = "x" * (MAX_FETCH_CHARS + 500)
    mock_client.extract.return_value = {
        "results": [{"url": "https://example.com/long", "raw_content": long_text}],
        "failed_results": [],
    }
    mock_get_client.return_value = mock_client

    result = tool_fetch("https://example.com/long")

    assert len(result.text) == MAX_FETCH_CHARS


def _fake_tool_use_response(input_dict: dict):
    """Build a fake Anthropic Message with a single tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.input = input_dict
    response = MagicMock()
    response.content = [block]
    return response


VALID_COMPARISON_INPUT = {
    "stance": "refutes",
    "confidence": 0.85,
    "quote": "There is no evidence linking the measles vaccine to autism.",
    "reasoning": "The text directly contradicts the claim.",
}


@patch("misinfo_agent.tools._get_anthropic_client")
def test_tool_compare_claim_to_text_returns_structured_result(mock_get_client):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_tool_use_response(VALID_COMPARISON_INPUT)
    mock_get_client.return_value = mock_client

    result = tool_compare_claim_to_text(
        claim="The measles vaccine causes autism.",
        text="There is no evidence linking the measles vaccine to autism.",
    )

    assert isinstance(result, ComparisonResult)
    assert result.stance == "refutes"
    assert result.confidence == 0.85
    assert result.quote == "There is no evidence linking the measles vaccine to autism."


@patch("misinfo_agent.tools._get_anthropic_client")
def test_tool_compare_claim_to_text_forces_tool_choice(mock_get_client):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _fake_tool_use_response(VALID_COMPARISON_INPUT)
    mock_get_client.return_value = mock_client

    tool_compare_claim_to_text(claim="some claim", text="some text")

    _, kwargs = mock_client.messages.create.call_args
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_comparison"}
    assert kwargs["tools"][0]["name"] == "record_comparison"


@patch("misinfo_agent.tools._get_anthropic_client")
def test_tool_compare_claim_to_text_retries_transient_errors(mock_get_client):
    mock_client = MagicMock()
    fake_request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    mock_client.messages.create.side_effect = [
        APIConnectionError(request=fake_request),
        _fake_tool_use_response(VALID_COMPARISON_INPUT),
    ]
    mock_get_client.return_value = mock_client

    result = tool_compare_claim_to_text(claim="some claim", text="some text")

    assert result.stance == "refutes"
    assert mock_client.messages.create.call_count == 2


@patch("misinfo_agent.tools._get_anthropic_client")
def test_tool_compare_claim_to_text_does_not_retry_non_transient_errors(mock_get_client):
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = ValueError("bad request shape")
    mock_get_client.return_value = mock_client

    with pytest.raises(ValueError):
        tool_compare_claim_to_text(claim="some claim", text="some text")

    assert mock_client.messages.create.call_count == 1


FAKE_CREDIBILITY_TABLE = {
    "cdc.gov": {
        "domain": "cdc.gov",
        "name": "Centers for Disease Control and Prevention",
        "type": "government",
        "credibility": "high",
        "bias": "center",
    },
    "example-blog.com": {
        "domain": "example-blog.com",
        "name": "Example Blog",
        "type": "blog",
        "credibility": "low",
        "bias": "right",
    },
}


def test_extract_domain_strips_scheme_and_www():
    assert _extract_domain("https://www.cdc.gov/measles/about.html") == "cdc.gov"


def test_extract_domain_keeps_non_www_subdomain():
    assert _extract_domain("https://wonder.cdc.gov/data") == "wonder.cdc.gov"


def test_extract_domain_lowercases():
    assert _extract_domain("https://CDC.GOV/page") == "cdc.gov"


@patch("misinfo_agent.tools._load_credibility_table")
def test_tool_assess_source_known_domain_returns_row(mock_load_table):
    mock_load_table.return_value = FAKE_CREDIBILITY_TABLE

    result = tool_assess_source("https://www.cdc.gov/measles/about.html")

    assert isinstance(result, SourceAssessment)
    assert result.domain == "cdc.gov"
    assert result.name == "Centers for Disease Control and Prevention"
    assert result.type == "government"
    assert result.credibility == "high"
    assert result.bias == "center"


@patch("misinfo_agent.tools._load_credibility_table")
def test_tool_assess_source_unknown_domain_returns_unknown_defaults(mock_load_table):
    mock_load_table.return_value = FAKE_CREDIBILITY_TABLE

    result = tool_assess_source("https://some-random-site.example/page")

    assert result.domain == "some-random-site.example"
    assert result.name is None
    assert result.type == "unknown"
    assert result.credibility == "unknown"
    assert result.bias == "unknown"


@patch("misinfo_agent.tools._load_credibility_table")
def test_tool_assess_source_strips_www_before_lookup(mock_load_table):
    mock_load_table.return_value = FAKE_CREDIBILITY_TABLE

    result = tool_assess_source("http://example-blog.com/post/1")

    assert result.credibility == "low"
    assert result.bias == "right"


FAKE_TABLE_WITH_FACT_CHECKERS = {
    "politifact.com": {
        "domain": "politifact.com", "name": "PolitiFact", "type": "fact-checker",
        "credibility": "high", "bias": "center",
    },
    "snopes.com": {
        "domain": "snopes.com", "name": "Snopes", "type": "fact-checker",
        "credibility": "high", "bias": "center",
    },
    "cdc.gov": {
        "domain": "cdc.gov", "name": "CDC", "type": "government",
        "credibility": "high", "bias": "center",
    },
}


@patch("misinfo_agent.tools._load_credibility_table")
def test_fact_checker_domains_returns_only_fact_checker_type(mock_load_table):
    mock_load_table.return_value = FAKE_TABLE_WITH_FACT_CHECKERS

    assert fact_checker_domains() == {"politifact.com", "snopes.com"}

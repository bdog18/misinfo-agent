"""Phase 2: tool_search, tool_fetch, tool_assess_source, tool_compare_claim_to_text."""

import csv
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from anthropic import (
    Anthropic,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OverloadedError,
    RateLimitError,
)
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

# ---------------------------------------------------------------------------
# Tavily client setup (shared by tool_search, tool_fetch)
# ---------------------------------------------------------------------------

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    """Lazily construct the shared TavilyClient, reading TAVILY_API_KEY from the environment.

    Tests should monkeypatch/mock this function rather than hit the real API.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = TavilyClient(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# tool_search
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """One search hit, reshaped from Tavily's response into the agent's own schema."""

    title: str
    url: str
    snippet: str
    published_date: str | None = None
    score: float


def tool_search(
    query: str, max_results: int = 5, *, exclude_domains: set[str] | None = None
) -> list[SearchResult]:
    """Search the web for a query via Tavily and return structured results.

    Parameters
    ----------
    query: search string built by the agent (e.g. the claim text, or a
        follow-up query like "site:cdc.gov measles vaccine autism").
    max_results: cap on number of results returned.
    exclude_domains: domains to drop from the results before returning,
        e.g. fact_checker_domains() during an eval run — this eval's claim
        set is scraped from fact-checkers' own articles, so leaving them
        searchable lets the agent find and cite the answer key directly
        instead of investigating independently. Not exposed to the model
        as a tool parameter; the caller (agent/baseline loop) injects it.

    Returns
    -------
    A list of SearchResult, ordered by Tavily's relevance score (highest
    first, since Tavily returns them pre-sorted). Empty list if the query
    returns no hits.

    Raises
    ------
    RuntimeError if TAVILY_API_KEY is not configured.
    Any exception Tavily's client raises on request failure (rate limit,
    network error, bad key) propagates to the caller as-is — the agent loop
    is responsible for catching it and logging a TraceStep, not this
    function.
    """
    client = _get_client()

    response = client.search(query, max_results=max_results)
    results = [SearchResult(
        title=result["title"],
        url=result["url"],
        snippet=result["content"],
        published_date=result.get("published_date"),
        score=result["score"]
    ) for result in response["results"]]

    if exclude_domains:
        results = [r for r in results if _extract_domain(r.url) not in exclude_domains]

    return results


# ---------------------------------------------------------------------------
# tool_fetch
# ---------------------------------------------------------------------------

MAX_FETCH_CHARS = 8000


class FetchResult(BaseModel):
    """Extracted text for a single URL, or the reason extraction failed."""

    url: str
    text: str | None = None
    error: str | None = None


def tool_fetch(url: str) -> FetchResult:
    """Fetch and extract the readable text of a single URL via Tavily's extract endpoint.

    Parameters
    ----------
    url: the page to fetch — typically a `url` from a SearchResult the
        agent decided to look at more closely.

    Returns
    -------
    A FetchResult. On success, `text` holds the extracted content (capped
    at MAX_FETCH_CHARS characters) and `error` is None. On failure (404,
    timeout, blocked, extraction failure), `text` is None and `error` holds
    a human-readable reason — this function does NOT raise for a single
    dead/blocked url, since hitting one is a routine, expected outcome
    during an investigation, unlike a Tavily request-level failure.

    Raises
    ------
    Any exception Tavily's client raises for a request-level failure that
    isn't about this specific url (e.g. bad API key, rate limit) still
    propagates to the caller, same as tool_search.
    """
    client = _get_client()

    response = client.extract(urls=[url])

    match = next((r for r in response["results"] if r["url"] == url), None)
    if match is not None:
        return FetchResult(url=url, text=match["raw_content"][:MAX_FETCH_CHARS])

    failure = next((f for f in response["failed_results"] if f["url"] == url), None)
    if failure is not None:
        return FetchResult(url=url, error=failure["error"])

    return FetchResult(url=url, error="Tavily returned no result for this url")


# ---------------------------------------------------------------------------
# Anthropic client setup (shared by tool_compare_claim_to_text)
# ---------------------------------------------------------------------------

HAIKU_MODEL = "claude-haiku-4-5-20251001"

_anthropic_client: Anthropic | None = None


def _get_anthropic_client() -> Anthropic:
    """Lazily construct the shared Anthropic client, reading ANTHROPIC_API_KEY from the environment.

    Tests should monkeypatch/mock this function rather than hit the real API.
    """
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


# ---------------------------------------------------------------------------
# tool_compare_claim_to_text
# ---------------------------------------------------------------------------


class ComparisonResult(BaseModel):
    """Haiku's structured judgment of one piece of source text against one claim."""

    stance: Literal["supports", "refutes", "irrelevant", "mixed"]
    confidence: float = Field(ge=0.0, le=1.0)
    quote: str | None = Field(
        default=None,
        description="Verbatim quote from the text backing this stance, or None if irrelevant",
    )
    reasoning: str


_COMPARE_TOOL_NAME = "record_comparison"

_COMPARE_TOOL = {
    "name": _COMPARE_TOOL_NAME,
    "description": (
        "Record a structured judgment of whether a piece of source text "
        "supports, refutes, is irrelevant to, or offers mixed evidence "
        "about a claim, based strictly on what the text itself literally "
        "states — not on outside knowledge about the source's reliability."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "stance": {
                "type": "string",
                "enum": ["supports", "refutes", "irrelevant", "mixed"],
            },
            "confidence": {
                "type": "number",
                "description": "0.0-1.0 confidence in this stance judgment.",
            },
            "quote": {
                "type": ["string", "null"],
                "description": (
                    "Verbatim quote from the text that backs this stance, "
                    "or null if stance is irrelevant."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One or two sentence rationale for the stance.",
            },
        },
        "required": ["stance", "confidence", "reasoning"],
    },
}

# The high-frequency comparison call is more likely to hit a transient
# rate limit / overload than the occasional Tavily call, so retry a few
# times here rather than pushing that handling onto every call site. Only
# retry errors that are actually transient — not e.g. AuthenticationError
# or BadRequestError, which will never succeed on a retry.
_TRANSIENT_ANTHROPIC_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    OverloadedError,
)


@retry(
    retry=retry_if_exception_type(_TRANSIENT_ANTHROPIC_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=8),
)
def tool_compare_claim_to_text(claim: str, text: str) -> ComparisonResult:
    """Judge whether a piece of fetched source text supports/refutes a claim, via Haiku.

    This is the high-frequency, cheap primitive the ReAct loop calls once
    per source it reads — it should NOT run on the orchestrator model.

    Parameters
    ----------
    claim: the claim text being investigated.
    text: fetched source text to judge against the claim (e.g. a
        FetchResult.text).

    Returns
    -------
    A ComparisonResult with Haiku's stance, confidence, the quote it based
    that stance on, and a short reasoning string. Stance is judged strictly
    from the literal content of `text` — Haiku is explicitly instructed to
    ignore whatever it separately knows about the source's reliability or
    reputation (that's tool_assess_source's job), so a text reporting a
    since-debunked study's own claim should be judged by what that claim
    asserts, not by outside knowledge of how it was later received.

    Raises
    ------
    RuntimeError if ANTHROPIC_API_KEY is not configured.
    Transient Anthropic errors (rate limit, overload, timeout, connection,
    5xx) are retried up to 3 times with exponential backoff before
    propagating. Non-transient errors (bad key, bad request) propagate
    immediately.
    """
    
    client = _get_anthropic_client()
    
    response = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=1024,
        tools=[_COMPARE_TOOL],
        tool_choice={"type": "tool", "name": _COMPARE_TOOL_NAME},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Claim: {claim}\n\n"
                    f"Source text: {text}\n\n"
                    "Judge stance based ONLY on what the source text above "
                    "literally states — not on anything you separately know "
                    "about the source's general reliability, reputation, or "
                    "whether it was later validated or debunked. If the text "
                    "reports that a study/person claimed something, judge "
                    "the stance based on what is being reported, not on "
                    "outside knowledge of how that claim was later received. "
                    "Source reliability is judged by a different tool — your "
                    "job here is purely: what does this text say?"
                ),
            }
        ]
    )
    return ComparisonResult(**response.content[0].input)


# ---------------------------------------------------------------------------
# tool_assess_source
# ---------------------------------------------------------------------------

# The credibility/bias table's methodology is documented in the README —
# see the "Source credibility table" section for provenance and caveats.
CREDIBILITY_TABLE_PATH = Path(__file__).resolve().parent / "data" / "source_credibility.csv"

SourceType = Literal[
    "government",
    "academic",
    "fact-checker",
    "news",
    "state-media",
    "research",
    "media-watchdog",
    "partisan-outlet",
    "blog",
    "satire",
    "unknown",
]
Credibility = Literal["high", "medium", "low", "unknown"]
Bias = Literal["left", "center-left", "center", "center-right", "right", "unknown"]


class SourceAssessment(BaseModel):
    """Credibility/bias rating for one domain, looked up from the curated CSV."""

    domain: str
    name: str | None = None
    type: SourceType = "unknown"
    credibility: Credibility = "unknown"
    bias: Bias = "unknown"


_credibility_table: dict[str, dict] | None = None


def _load_credibility_table() -> dict[str, dict]:
    """Lazily load and cache the CSV into a dict keyed by domain.

    Tests should monkeypatch this function (or _credibility_table directly)
    rather than depend on the real CSV's contents.
    """
    global _credibility_table
    if _credibility_table is None:
        table = {}
        with CREDIBILITY_TABLE_PATH.open(newline="") as f:
            for row in csv.DictReader(f):
                table[row["domain"].strip().lower()] = row
        _credibility_table = table
    return _credibility_table


def _extract_domain(url: str) -> str:
    """Extract a lowercase, www.-stripped domain from a url.

    e.g. "https://www.cdc.gov/measles/about.html" -> "cdc.gov"
         "https://wonder.cdc.gov/data" -> "wonder.cdc.gov" (kept as-is —
         only a leading "www." is stripped, no further subdomain collapsing)
    """
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[len("www.") :]
    return netloc


def fact_checker_domains() -> set[str]:
    """Domains rated type="fact-checker" in the curated credibility table.

    Used as tool_search's exclude_domains during eval runs — see
    tool_search's docstring for why.
    """
    table = _load_credibility_table()
    return {domain for domain, row in table.items() if row["type"] == "fact-checker"}


def tool_assess_source(url: str) -> SourceAssessment:
    """Look up a url's domain in the curated credibility/bias table.

    Parameters
    ----------
    url: a url the agent is considering as a source — typically a `url`
        from a SearchResult or FetchResult.

    Returns
    -------
    A SourceAssessment. If the domain (after stripping a leading "www.")
    matches a row in the CSV, its name/type/credibility/bias are returned.
    If there's no match, a SourceAssessment with all "unknown" fields is
    returned — this function never raises for an unrated domain, since most
    domains the agent encounters won't be in a hand-curated ~50-100 row
    table.
    """
    domain = _extract_domain(url)
    table = _load_credibility_table()
    row = table.get(domain)
    
    if row is not None:
        return SourceAssessment(**row)
    else:
        return SourceAssessment(domain=domain)
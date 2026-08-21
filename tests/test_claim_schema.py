import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from misinfo_agent.eval.schema import POLITIFACT_VERDICT_MAP, Citation, Claim

CLAIMS_PATH = Path(__file__).resolve().parent.parent / "misinfo_agent" / "eval" / "claims.jsonl"


def _minimal_claim(**overrides) -> dict:
    base = dict(
        id="test-0001",
        source="politifact",
        claim="This is a sufficiently long test claim statement.",
        verdict="true",
        verdict_detail="True",
        fact_check_url="https://politifact.com/factchecks/2026/jan/01/someone/some-claim/",
    )
    base.update(overrides)
    return base


def test_minimal_claim_is_valid():
    claim = Claim(**_minimal_claim())
    assert claim.verdict == "true"
    assert claim.reviewed is False
    assert claim.citations == []


def test_claim_text_too_short_rejected():
    with pytest.raises(ValidationError):
        Claim(**_minimal_claim(claim="too short"))


def test_invalid_verdict_rejected():
    with pytest.raises(ValidationError):
        Claim(**_minimal_claim(verdict="unclear"))


def test_invalid_fact_check_url_rejected():
    with pytest.raises(ValidationError):
        Claim(**_minimal_claim(fact_check_url="not-a-url"))


def test_citation_round_trips():
    claim = Claim(
        **_minimal_claim(
            citations=[
                Citation(
                    publisher="CDC",
                    title="Measles vaccine safety",
                    url="https://www.cdc.gov/measles/about/questions.html",
                    cited_date="Aug. 5, 2026",
                )
            ]
        )
    )
    assert claim.citations[0].publisher == "CDC"


def test_politifact_verdict_map_covers_all_ratings():
    expected_slugs = {"true", "mostly-true", "half-true", "mostly-false", "false", "pants-fire"}
    assert set(POLITIFACT_VERDICT_MAP) == expected_slugs
    assert set(POLITIFACT_VERDICT_MAP.values()) == {"true", "mixed", "false"}


def test_half_true_maps_to_mixed_not_false():
    assert POLITIFACT_VERDICT_MAP["half-true"] == "mixed"


def test_mostly_false_maps_to_false_not_mixed():
    assert POLITIFACT_VERDICT_MAP["mostly-false"] == "false"


@pytest.mark.skipif(not CLAIMS_PATH.exists(), reason="claims.jsonl not generated yet")
def test_claims_jsonl_entries_all_conform_to_schema():
    with CLAIMS_PATH.open() as f:
        lines = [line for line in f if line.strip()]
    assert lines, "claims.jsonl is empty"
    ids = set()
    for line in lines:
        data = json.loads(line)
        claim = Claim(**data)
        assert claim.id not in ids, f"duplicate id {claim.id}"
        ids.add(claim.id)

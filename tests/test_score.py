import dataclasses
import json

from misinfo_agent.eval.schema import Citation, Claim
from misinfo_agent.eval.score import (
    attribute_failure,
    found_disconfirming_evidence,
    score_investigation,
    score_run,
    summarize,
)
from misinfo_agent.trace import Evidence, Investigation


def _claim(verdict="false", citation_urls=("https://www.cdc.gov/measles/about.html",)) -> Claim:
    return Claim(
        id="politifact-0001",
        source="PolitiFact",
        claim="The measles vaccine causes autism.",
        verdict=verdict,
        verdict_detail="False",
        fact_check_url="https://www.politifact.com/factchecks/example/",
        citations=[Citation(publisher="CDC", title="Measles", url=u) for u in citation_urls],
        reviewed=True,
    )


def _investigation_dict(*, verdict="false", evidence_urls=(), arm="agent") -> dict:
    """Build a real Investigation and asdict() it, so the fixture matches
    exactly what harness.py's dataclasses.asdict(investigation) produces —
    not a hand-rolled dict that could drift from the real schema.
    """
    inv = Investigation(claim="The measles vaccine causes autism.", arm=arm, model="claude-sonnet-5")
    inv.verdict = verdict
    for i, url in enumerate(evidence_urls):
        inv.add_evidence(
            Evidence(
                url=url, stance="refutes", confidence=0.9, quote=None,
                source_credibility="high", source_bias="center", step_number=i + 1,
            )
        )
    inv.add_step(
        thought="t", action="tool_search", action_input={}, observation="o",
        input_tokens=100, output_tokens=20,
    )
    return dataclasses.asdict(inv)


def test_found_disconfirming_evidence_matches_on_domain_not_exact_url():
    claim = _claim(citation_urls=["https://www.cdc.gov/measles/about.html"])
    inv = _investigation_dict(evidence_urls=["https://cdc.gov/measles/faq.html"])  # different page, same domain

    assert found_disconfirming_evidence(inv, claim) is True


def test_found_disconfirming_evidence_false_when_no_domain_overlap():
    claim = _claim(citation_urls=["https://www.cdc.gov/measles/about.html"])
    inv = _investigation_dict(evidence_urls=["https://example.com/unrelated"])

    assert found_disconfirming_evidence(inv, claim) is False


def test_found_disconfirming_evidence_false_when_claim_has_no_citations():
    claim = _claim(citation_urls=[])
    inv = _investigation_dict(evidence_urls=["https://cdc.gov/measles/faq.html"])

    assert found_disconfirming_evidence(inv, claim) is False


def test_attribute_failure_none_when_verdict_correct():
    claim = _claim(verdict="false")
    inv = _investigation_dict(verdict="false", evidence_urls=[])

    assert attribute_failure(inv, claim) is None


def test_attribute_failure_reasoning_failure_when_wrong_but_found_the_source():
    claim = _claim(verdict="false", citation_urls=["https://www.cdc.gov/measles/about.html"])
    inv = _investigation_dict(verdict="true", evidence_urls=["https://cdc.gov/measles/faq.html"])

    assert attribute_failure(inv, claim) == "reasoning_failure"


def test_attribute_failure_search_failure_when_wrong_and_never_found_the_source():
    claim = _claim(verdict="false", citation_urls=["https://www.cdc.gov/measles/about.html"])
    inv = _investigation_dict(verdict="true", evidence_urls=["https://example.com/unrelated"])

    assert attribute_failure(inv, claim) == "search_failure"


def test_score_investigation_fields():
    claim = _claim(verdict="false", citation_urls=["https://www.cdc.gov/measles/about.html"])
    inv = _investigation_dict(verdict="false", evidence_urls=["https://cdc.gov/measles/faq.html"])

    row = score_investigation(inv, claim)

    assert row["claim_id"] == "politifact-0001"
    assert row["arm"] == "agent"
    assert row["correct"] is True
    assert row["predicted_verdict"] == "false"
    assert row["ground_truth_verdict"] == "false"
    assert row["sought_disconfirming_evidence"] is True
    assert row["found_disconfirming_evidence"] is True
    assert row["failure_attribution"] is None
    assert row["step_count"] == 1
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 20


def test_summarize_aggregates_per_arm():
    rows = [
        {"arm": "agent", "correct": True, "sought_disconfirming_evidence": True,
         "failure_attribution": None, "step_count": 4, "input_tokens": 1000, "output_tokens": 100},
        {"arm": "agent", "correct": False, "sought_disconfirming_evidence": False,
         "failure_attribution": "search_failure", "step_count": 6, "input_tokens": 2000, "output_tokens": 200},
        {"arm": "baseline", "correct": True, "sought_disconfirming_evidence": False,
         "failure_attribution": None, "step_count": 2, "input_tokens": 500, "output_tokens": 50},
    ]

    summary = summarize(rows)

    assert summary["agent"]["n_claims"] == 2
    assert summary["agent"]["accuracy"] == 0.5
    assert summary["agent"]["evidence_seeking_rate"] == 0.5
    assert summary["agent"]["avg_step_count"] == 5
    assert summary["agent"]["n_wrong"] == 1
    assert summary["agent"]["n_search_failure"] == 1
    assert summary["agent"]["n_reasoning_failure"] == 0

    assert summary["baseline"]["n_claims"] == 1
    assert summary["baseline"]["accuracy"] == 1.0


def test_score_run_skips_error_records_and_unknown_claims(tmp_path):
    claim = _claim(verdict="false")
    inv = _investigation_dict(verdict="false", evidence_urls=[])

    results_path = tmp_path / "results.jsonl"
    lines = [
        {"claim_id": claim.id, "arm": "agent", "investigation": inv},
        {"claim_id": claim.id, "arm": "baseline", "error": "RuntimeError: boom"},
        {"claim_id": "unknown-claim", "arm": "agent", "investigation": inv},
    ]
    results_path.write_text("\n".join(json.dumps(l) for l in lines) + "\n")

    summary = score_run(results_path, [claim])

    assert "baseline" not in summary
    assert summary["agent"]["n_claims"] == 1
    assert summary["agent"]["accuracy"] == 1.0

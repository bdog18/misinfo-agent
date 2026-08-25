import json
from unittest.mock import patch

from misinfo_agent.eval import harness
from misinfo_agent.eval.schema import Claim
from misinfo_agent.trace import Investigation


def _claim(id="politifact-0001", verdict="false", reviewed=True) -> Claim:
    return Claim(
        id=id,
        source="PolitiFact",
        claim="The measles vaccine causes autism.",
        verdict=verdict,
        verdict_detail="False",
        fact_check_url="https://www.politifact.com/factchecks/example/",
        reviewed=reviewed,
    )


def _write_claims_jsonl(path, claims: list[Claim]):
    path.write_text("\n".join(c.model_dump_json() for c in claims) + "\n")


def _investigation(arm: str, claim_text: str) -> Investigation:
    inv = Investigation(claim=claim_text, arm=arm, model="claude-sonnet-5")
    inv.verdict = "false"
    inv.confidence = 0.9
    inv.reasoning = "because"
    inv.stop_reason = "verdict_submitted"
    return inv


def test_load_claims_filters_unreviewed_by_default(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    _write_claims_jsonl(claims_path, [_claim("c1", reviewed=True), _claim("c2", reviewed=False)])

    claims = harness.load_claims(claims_path)

    assert [c.id for c in claims] == ["c1"]


def test_load_claims_include_unreviewed(tmp_path):
    claims_path = tmp_path / "claims.jsonl"
    _write_claims_jsonl(claims_path, [_claim("c1", reviewed=True), _claim("c2", reviewed=False)])

    claims = harness.load_claims(claims_path, reviewed_only=False)

    assert {c.id for c in claims} == {"c1", "c2"}


def test_run_batch_writes_one_record_per_claim_per_arm(tmp_path, monkeypatch):
    monkeypatch.setitem(harness.ARMS, "agent", lambda claim_text: _investigation("agent", claim_text))
    monkeypatch.setitem(harness.ARMS, "baseline", lambda claim_text: _investigation("baseline", claim_text))

    results_path = tmp_path / "results.jsonl"
    harness.run_batch([_claim("c1"), _claim("c2")], results_path)

    records = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert {(r["claim_id"], r["arm"]) for r in records} == {
        ("c1", "agent"), ("c1", "baseline"), ("c2", "agent"), ("c2", "baseline"),
    }
    assert all("investigation" in r for r in records)


def test_run_batch_records_error_without_aborting(tmp_path, monkeypatch):
    def failing_agent(claim_text):
        raise RuntimeError("boom")

    monkeypatch.setitem(harness.ARMS, "agent", failing_agent)
    monkeypatch.setitem(harness.ARMS, "baseline", lambda claim_text: _investigation("baseline", claim_text))

    results_path = tmp_path / "results.jsonl"
    harness.run_batch([_claim("c1")], results_path)

    records = {(r["claim_id"], r["arm"]): r for r in (json.loads(l) for l in results_path.read_text().splitlines())}
    assert "RuntimeError: boom" in records[("c1", "agent")]["error"]
    assert "investigation" in records[("c1", "baseline")]


def test_run_batch_skips_already_completed_pairs(tmp_path, monkeypatch):
    calls = []

    def fake_agent(claim_text):
        calls.append(claim_text)
        return _investigation("agent", claim_text)

    monkeypatch.setitem(harness.ARMS, "agent", fake_agent)
    monkeypatch.setitem(harness.ARMS, "baseline", fake_agent)

    results_path = tmp_path / "results.jsonl"
    results_path.write_text(json.dumps({"claim_id": "c1", "arm": "agent"}) + "\n")

    harness.run_batch([_claim("c1")], results_path, arms=("agent", "baseline"))

    records = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert len(records) == 2  # the pre-existing line + only the new "baseline" run
    assert len(calls) == 1


def test_new_run_path_is_timestamped_under_base_dir(tmp_path):
    path = harness.new_run_path(tmp_path)

    assert path.parent == tmp_path
    assert path.suffix == ".jsonl"


@patch("misinfo_agent.tools.fact_checker_domains")
@patch("misinfo_agent.agent.run_investigation")
def test_arms_agent_passes_fact_checker_domains_as_exclude(mock_run_investigation, mock_fact_checker_domains):
    mock_fact_checker_domains.return_value = {"politifact.com"}
    mock_run_investigation.return_value = _investigation("agent", "c")

    harness.ARMS["agent"]("some claim")

    mock_run_investigation.assert_called_once_with("some claim", exclude_domains={"politifact.com"})


@patch("misinfo_agent.tools.fact_checker_domains")
@patch("misinfo_agent.baseline.run_baseline")
def test_arms_baseline_passes_fact_checker_domains_as_exclude(mock_run_baseline, mock_fact_checker_domains):
    mock_fact_checker_domains.return_value = {"politifact.com"}
    mock_run_baseline.return_value = _investigation("baseline", "c")

    harness.ARMS["baseline"]("some claim")

    mock_run_baseline.assert_called_once_with("some claim", exclude_domains={"politifact.com"})

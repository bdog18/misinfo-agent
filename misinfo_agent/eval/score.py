"""Phase 5: accuracy, evidence-seeking rate, failure attribution."""

import json
from pathlib import Path
from typing import Literal

from misinfo_agent import tools
from misinfo_agent.eval.harness import load_claims
from misinfo_agent.eval.schema import Claim


def found_disconfirming_evidence(investigation: dict, claim: Claim) -> bool:
    """Check if the investigation found any disconfirming evidence for the claim."""
    citation_domains = {tools._extract_domain(c.url) for c in claim.citations}
    evidence_domains = {tools._extract_domain(e["url"]) for e in investigation["evidence"]}
    return bool(citation_domains & evidence_domains)


def attribute_failure(
    investigation: dict, claim: Claim
) -> Literal["reasoning_failure", "search_failure", "incomplete"] | None:
    """Attribute a wrong verdict to reasoning, search, or incompleteness.

    "incomplete" (the investigation hit max_steps without ever submitting a
    verdict) is checked first and kept distinct from reasoning/search
    failure - a None verdict never actually concluded anything, so labeling
    it "reasoning_failure" just because it happened to have citation-matching
    evidence sitting in its list when it ran out of steps would misrepresent
    what went wrong.
    """
    if investigation["verdict"] == claim.verdict:
        return None
    if investigation["stop_reason"] == "max_steps_reached":
        return "incomplete"
    if found_disconfirming_evidence(investigation, claim):
        return "reasoning_failure"
    return "search_failure"


def score_investigation(investigation: dict, claim: Claim) -> dict:
    """Score a single investigation against the ground truth claim."""
    return {
        "claim_id": claim.id,
        "arm": investigation["arm"],
        "correct": investigation["verdict"] == claim.verdict,
        "predicted_verdict": investigation["verdict"],
        "ground_truth_verdict": claim.verdict,
        "sought_disconfirming_evidence": any(investigation["evidence"][i]["stance"] in ("refutes", "mixed") for i in range(len(investigation["evidence"]))),
        "found_disconfirming_evidence": found_disconfirming_evidence(investigation, claim),
        "failure_attribution": attribute_failure(investigation, claim),
        "step_count": len(investigation["steps"]),
        "input_tokens": sum(investigation["steps"][i]["input_tokens"] for i in range(len(investigation["steps"])) if "input_tokens" in investigation["steps"][i]) if investigation["steps"] else 0,
        "output_tokens": sum(investigation["steps"][i]["output_tokens"] for i in range(len(investigation["steps"])) if "output_tokens" in investigation["steps"][i]) if investigation["steps"] else 0,
    }


def summarize(rows: list[dict]) -> dict:
    """Summarize the results of a run of investigations."""
    summary = {}
    for arm in set(row["arm"] for row in rows):
        arm_rows = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            "n_claims": len(arm_rows),
            "accuracy": sum(row["correct"] for row in arm_rows) / len(arm_rows),
            "evidence_seeking_rate": sum(row["sought_disconfirming_evidence"] for row in arm_rows) / len(arm_rows),
            "avg_step_count": sum(row["step_count"] for row in arm_rows) / len(arm_rows),
            "avg_input_tokens": sum(row["input_tokens"] for row in arm_rows) / len(arm_rows),
            "avg_output_tokens": sum(row["output_tokens"] for row in arm_rows) / len(arm_rows),
            "n_wrong": sum(not row["correct"] for row in arm_rows),
            "n_reasoning_failure": sum(row["failure_attribution"] == "reasoning_failure" for row in arm_rows),
            "n_search_failure": sum(row["failure_attribution"] == "search_failure" for row in arm_rows),
            "n_incomplete": sum(row["failure_attribution"] == "incomplete" for row in arm_rows),
        }
    return summary


def score_run(results_path: Path, claims: list[Claim]) -> dict:
    """Score a run of investigations against the ground truth claims."""
    rows = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("error"):
                continue  # Skip scoring if there was an error in the investigation
            
            claim = next((c for c in claims if c.id == record["claim_id"]), None)
            if claim is None:
                continue  # Skip scoring if the claim is not found
            
            rows.append(score_investigation(record["investigation"], claim))
    return summarize(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="path to a harness.py results JSONL file")
    parser.add_argument("--claims", type=Path, default=Path("misinfo_agent/eval/claims.jsonl"))
    args = parser.parse_args()

    claim_set = load_claims(args.claims)
    summary = score_run(args.results, claim_set)
    print(json.dumps(summary, indent=2))
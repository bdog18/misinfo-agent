"""Phase 5: batch runner, both arms."""

import dataclasses
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path

from misinfo_agent import agent, baseline, tools
from misinfo_agent.eval.schema import Claim

# fact_checker_domains() is called fresh per invocation (cheap - it just
# re-reads the already-cached credibility table) rather than baked in once
# at import time, so eval runs always reflect the table's current contents
# and tests can patch it without needing to reimport this module.
ARMS = {
    "agent": lambda claim_text: agent.run_investigation(
        claim_text, exclude_domains=tools.fact_checker_domains()
    ),
    "baseline": lambda claim_text: baseline.run_baseline(
        claim_text, exclude_domains=tools.fact_checker_domains()
    ),
}

def load_claims(claims_path: Path, *, reviewed_only=True) -> list[Claim]:
    """Load claims from a JSONL file."""

    claim_list = []
    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                claim = Claim.model_validate_json(line)
                if reviewed_only and claim.reviewed is False:
                    continue
                claim_list.append(claim)
            except Exception as e:
                print(f"Error validating claim: {e}")
                raise
        
    return claim_list


def run_batch(
    claims: list[Claim], 
    results_path: Path, 
    *, 
    arms: tuple[str, ...] = ("agent", "baseline")
) -> Path:
    """Run a batch of claims through the specified arms and write results to a JSONL file.
    Each claim will be processed by each arm, and the results will be written to the specified path."""
    
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    done = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                record = json.loads(line)
                done.add((record["claim_id"], record["arm"]))
    
    with open(results_path, "a") as out:
        for claim in claims:
            for arm in arms:
                if arm not in ARMS:
                    raise ValueError(f"Unknown arm: {arm}")
                if (claim.id, arm) in done:
                    continue
                try:
                    investigation = ARMS[arm](claim.claim)
                    out.write(json.dumps({
                        "claim_id": claim.id,
                        "arm": arm,
                        "investigation": dataclasses.asdict(investigation)
                    }) + "\n")
                except Exception as e:
                    out.write(json.dumps({
                        "claim_id": claim.id,
                        "arm": arm,
                        "error": f"{type(e).__name__}: {e}"
                    }) + "\n")
                finally:
                    out.flush()
                    
    return Path(results_path)


def new_run_path(base_dir: Path) -> Path:
    """Generate a new run path for storing results, based on the current date."""
    return Path(base_dir) / f"run_{datetime.now(ZoneInfo('America/Denver')).strftime('%Y_%m_%d_%H:%M:%S')}.jsonl"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, default=Path("misinfo_agent/eval/claims.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--arms", nargs="+", default=["agent", "baseline"], choices=list(ARMS))
    args = parser.parse_args()

    claim_set = load_claims(args.claims)
    if not claim_set:
        raise SystemExit(
            f"No reviewed claims found in {args.claims}. Review some first: "
            "python scripts/review_claims.py"
        )

    results_path = new_run_path(args.out_dir)
    print(f"Running {len(claim_set)} claim(s) x {len(args.arms)} arm(s) -> {results_path}")
    run_batch(claim_set, results_path, arms=tuple(args.arms))
    print(f"Done. Results at {results_path}")
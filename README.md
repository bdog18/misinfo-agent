# misinfo-agent

An LLM agent that investigates a claim: given a claim, it autonomously plans
its own fact-check — deciding what to search, which sources to check, when
it has enough evidence — and produces a verdict with a full reasoning trace.
Evaluated head-to-head against a single-shot RAG baseline (same tools, one
search, one answer, no loop) to answer one question: **does agentic
autonomy earn its cost here, or does it just add latency and expense
without matching quality?**

Status: in progress. This README will carry the results tables,
methodology, and findings once the eval harness has run. See
`PROJECT_STATUS` below for where things stand.

## Project status

- [x] Phase 1 — ground-truth claim schema + scraped batch (in review)
- [x] Phase 2 — tools (search, fetch, source credibility, claim comparison)
- [x] Phase 3 — ReAct agent loop
- [x] Phase 4 — single-shot baseline
- [~] Phase 5 — eval harness built (`misinfo_agent/eval/harness.py`,
  `score.py`) and verified against the real Anthropic/Tavily APIs; the
  batched run itself is held pending the claim set (see below)
- [x] Phase 6 — Gradio demo (`app/demo.py`) streams the agent's ReAct trace
  step by step alongside the single-shot baseline for the same claim, and
  the `Dockerfile` builds/runs it (verified locally); not yet deployed to a
  brendenrunion.com subdomain
- [ ] Phase 7 — writeup

## Ground-truth claim set

`misinfo_agent/eval/claims.jsonl` holds the eval set: real claims with a
PolitiFact ground-truth verdict and the actual citation trail the
fact-checker relied on. Schema and the verdict-bucketing methodology
(True/False/Mixed) are documented in `misinfo_agent/eval/schema.py`.

Currently 98 claims pulled with `scripts/scrape_politifact.py`, in range
for the 60-100 target but not yet balanced or reviewed: 81 false / 9 mixed
/ 8 true (82.7% false), and none marked `reviewed: true` yet. `reviewed:
false` marks entries not yet human-checked; nothing with `reviewed: false`
should be used for scoring, and the harness's `load_claims()` defaults to
`reviewed_only=True` (currently an empty set) so this can't happen by
accident. The skew toward `false` is worth correcting — or at least
explicitly caveating — before the real batched run, since it otherwise
mostly tests whether the model says "false," not disconfirming-evidence
seeking specifically.

`scripts/scrape_politifact.py` respects `politifact.com/robots.txt`
(`Crawl-delay: 10`) and is meant for small, occasional seed batches, not
bulk harvesting.

## Source credibility table

`misinfo_agent/data/source_credibility.csv` (~485 domains) backs
`tool_assess_source`'s credibility/bias/type lookup. The ratings were not
independently derived — they were compiled from two publicly available
rating resources, [Reality Team's credible sources
list](https://realityteam.org/resources/credible-sources/) and [Ad Fontes
Media's individual source rankings](https://adfontesmedia.com/rankings-by-individual-news-source/)
(the free rankings page, not Ad Fontes' paid full dataset), then parsed
into this project's schema with the help of an LLM. `credibility`
(high/medium/low), `bias` (left/center/right), and `type` reflect those
two organizations' own methodologies, not this project's independent
editorial judgment.

This is a deliberate scope trade-off for a portfolio project on a time
budget, not the original Phase 1 plan (which called for fully manual
curation to avoid exactly this kind of third-party-methodology
dependency). Anyone extending this table with new domains should treat
these two sources as reference material to weigh, not a canonical source
of truth — and should check each site's terms before pulling further data
in bulk.

## Eval harness

`misinfo_agent/eval/harness.py` runs the reviewed claim set through both
arms (agent, baseline), appending one JSON record per (claim, arm) to a
timestamped results file under `runs/` (gitignored). It's resumable — a
crashed or interrupted run picks back up rather than re-paying for work
already done — and a single claim/arm failure is caught and logged rather
than aborting the batch, since this is meant to run unattended.
`misinfo_agent/eval/score.py` scores a results file against ground truth:
accuracy, evidence-seeking rate, cost (steps/tokens), and per-wrong-verdict
failure attribution (reasoning failure vs. search failure, via domain-level
matching against the claim's citations).

**The batched run itself is intentionally not done yet** — see the claim
set skew/review status above.

### Fact-checker domain exclusion (and its limit)

Because this eval's claims are scraped from PolitiFact's own fact-checks, a
plain web search on a claim's exact wording often surfaces PolitiFact's own
conclusion directly — letting the agent cite the fact-checker's verdict as
its "evidence" instead of investigating independently. During eval runs
only (`misinfo_agent/eval/harness.py`'s `ARMS`), `tool_search` and
`tool_fetch` both hard-reject the domains rated `fact-checker` in the
credibility table (confirmed live: `tool_search` no longer returns them,
and `tool_fetch` is rejected the same way `tool_compare_claim_to_text` is
rejected for an unassessed source). This is eval-only, not built into the
tools themselves — a deployed fact-checking assistant should be allowed to
consult existing fact-checks as normal desk research; the leakage problem
is specific to this eval's claims and their own source overlapping.

This closes the direct route but not all of it: PolitiFact syndicates
fact-checks to partner outlets (e.g. a "PolitiFact Florida" piece
republished verbatim under a local NPR affiliate's own domain), so the same
conclusion can still turn up under a domain the credibility table has no
reason to tag as a fact-checker. A live-verified case: with the exclusion
in place, the agent independently found and cited a verbatim PolitiFact
Florida fact-check republished under `wlrn.org`. Domain-blocking narrows
the shortcut (that run took 10 steps and several fact-checker-name-specific
searches instead of finding it in 2) but can't close it completely by
construction — new syndication partners can't be enumerated in advance.
Treated as an accepted, documented limitation of using a public
fact-checking database as ground truth for a search-equipped agent, not a
bug: worth stating plainly in the eventual writeup rather than chasing an
ever-expanding blocklist.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,scrape]"
.venv/bin/pytest
```

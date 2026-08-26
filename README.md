# misinfo-agent

An LLM agent that investigates a claim: given a claim, it autonomously plans
its own fact-check — deciding what to search, which sources to check, when
it has enough evidence — and produces a verdict with a full reasoning trace.
Evaluated head-to-head against a single-shot RAG baseline (same tools, one
search, one answer, no loop) to answer one question: **does agentic
autonomy earn its cost here, or does it just add latency and expense
without matching quality?**

Status: Completed
Full writeup: [WRITEUP.md](WRITEUP.md)

## Project status

- [x] Phase 1 — ground-truth claim schema + scraped, reviewed batch
- [x] Phase 2 — tools (search, fetch, source credibility, claim comparison)
- [x] Phase 3 — ReAct agent loop
- [x] Phase 4 — single-shot baseline
- [x] Phase 5 — eval harness (`misinfo_agent/eval/harness.py`, `score.py`),
  batched run completed over 58 reviewed claims, both arms — see Results
- [x] Phase 6 — Gradio demo (`app/demo.py`) streams the agent's ReAct trace
  step by step alongside the single-shot baseline for the same claim, and
  the `Dockerfile` builds/runs it (verified locally); not yet deployed to a
  brendenrunion.com subdomain
- [x] Phase 7 — writeup

## Results

58 reviewed claims (19 true / 20 mixed / 19 false), both arms, real
Anthropic + Tavily API calls, run `runs/run_2026_08_25_19:33:39.jsonl`.
Full per-arm numbers via `python -m misinfo_agent.eval.score <results-file>`.

**Raw accuracy makes the agent look worse than the baseline — but that's
the wrong headline.** Baseline: 75.9% (44/58). Agent: 72.4% (42/58) raw.
The gap isn't reasoning quality: **19% of agent investigations (11/58)
never reached a verdict at all**, hitting the 15-step safety cap with
`verdict: null` — which scores as automatically wrong. Restricted to the
47 investigations that actually finished, the agent hits **89.4%
accuracy — well above the baseline's 75.9%.**

**Why investigations don't finish: the disconfirming-evidence guardrail
can starve on claims where it can't be satisfied.** `submit_verdict` hard-
rejects unless the agent has already gathered evidence that disconfirms,
complicates, or is mixed on the claim. For a claim that's actually *true*,
there may be no legitimate disconfirming evidence to find — so the agent
keeps searching, keeps getting rejected, and burns its whole step budget.
Stuck rate by ground truth: **true 26.3% (5/19) vs. mixed 15.0% (3/20) vs.
false 15.8% (3/19)** — a real, if modest-sample, skew toward the verdict
class where the guardrail's own assumption doesn't hold. This is reported
as a limitation of hard-enforced guardrails, not fixed and rerun — see
`score.py`'s `attribute_failure` for the "incomplete" category this
produced (kept separate from `reasoning_failure`/`search_failure`, since a
`null` verdict never actually concluded anything).

**Cost: the agent is ~34x more expensive per claim, unconditionally.**
Agent: ~129,142 input / ~4,229 output tokens/claim (~$0.30 at Sonnet 5
pricing). Baseline: ~2,614 input / ~365 output tokens/claim (~$0.009).
That multiple holds regardless of the accuracy story above — better
judgment when it completes, but nowhere near cheap.

**Failure attribution, for the 5 agent claims that finished with an
actually-wrong verdict:** 1 `reasoning_failure` (found the right source,
still concluded wrong), 4 `search_failure` (never found a source matching
the claim's citation domains). Baseline's 14 wrong claims are all
`search_failure` by construction — it never gathers tracked evidence at
all, so `found_disconfirming_evidence` is always `False` for it; this
isn't a finding about the baseline's reasoning, just the metric's
mechanical floor for an arm with no evidence-gathering step.

**Net answer to "does agentic autonomy earn its cost here":** better
judgment when the agent actually finishes, an unreliable completion rate
(especially on true claims, for a mechanistically understood reason), and
substantially more expensive either way. Not a clean win for autonomy.

## Ground-truth claim set

`misinfo_agent/eval/claims.jsonl` holds the eval set: real claims with a
PolitiFact ground-truth verdict and the actual citation trail the
fact-checker relied on. Schema and the verdict-bucketing methodology
(True/False/Mixed) are documented in `misinfo_agent/eval/schema.py`.

58 claims, all `reviewed: true`, pulled with `scripts/scrape_politifact.py`
via its `--rating` filter (backfilling `true`/`half-true` specifically
rather than only ever taking whatever's most recent in the general feed,
which skewed heavily `false`) and reviewed one-by-one with
`scripts/review_claims.py` (gitignored — a personal review tool, not
published methodology). Final split: 19 true / 20 mixed / 19 false —
genuinely balanced, not just in-range. `reviewed: false` marks entries not
yet human-checked; nothing with `reviewed: false` should be used for
scoring, and the harness's `load_claims()` defaults to `reviewed_only=True`
so this can't happen by accident.

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
failure attribution (`reasoning_failure`, `search_failure`, or
`incomplete` for an investigation that hit `max_steps` without ever
submitting a verdict — kept as its own category rather than folded into
the other two, since a `null` verdict never actually reasoned about
anything). See Results above for the actual numbers.

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

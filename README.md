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

- [x] Phase 1 — ground-truth claim schema + seed batch (in review)
- [x] Phase 2 — tools (search, fetch, source credibility, claim comparison)
- [ ] Phase 3 — ReAct agent loop
- [ ] Phase 4 — single-shot baseline
- [ ] Phase 5 — eval harness + batched run, both arms
- [ ] Phase 6 — Gradio demo (live trace) + Docker deploy
- [ ] Phase 7 — writeup

## Ground-truth claim set

`misinfo_agent/eval/claims.jsonl` holds the eval set: real claims with a
PolitiFact ground-truth verdict and the actual citation trail the
fact-checker relied on. Schema and the verdict-bucketing methodology
(True/False/Mixed) are documented in `misinfo_agent/eval/schema.py`.

A small seed batch (10 claims) was pulled with `scripts/scrape_politifact.py`
so the JSON format could be reviewed by hand before manually building out
the rest of the set to 60-100 claims, roughly balanced across
true/false/mixed. `reviewed: false` marks entries not yet human-checked;
nothing with `reviewed: false` should be used for scoring.

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

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,scrape]"
.venv/bin/pytest
```

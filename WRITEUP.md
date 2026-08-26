# Does an autonomous fact-checking agent earn its cost?

**Short answer: partially.** A ReAct-style agent that plans its own fact-check —
deciding what to search, when it has enough evidence, when to stop — was run
head-to-head against a single-shot RAG baseline (same tools, one search, one
answer, no loop) over 58 real claims with PolitiFact-sourced ground truth. On
raw accuracy the agent *loses*, 72.4% to 75.9%. That's the wrong headline: 19%
of agent investigations never reached a verdict at all, and a `null` verdict
scores as automatically wrong. Restricted to the 47 investigations that
actually finished, the agent hits 89.4% accuracy against the baseline's
75.9% — a real quality edge. But it costs about 34x more per claim regardless,
and the reason a fifth of investigations don't finish turns out to be a
structural property of one of the agent's own guardrails, not noise. Details
below.

## The question

The interesting question about agentic autonomy isn't "can it produce an
answer" — a single search-and-answer baseline can do that too, more cheaply.
It's whether letting a model plan its own multi-step investigation — choosing
what to search, following up on what it finds, deciding for itself when it
has enough — produces evidence-gathering and judgment that a single-shot
lookup can't. And specifically: does the extra freedom mean the agent
actually goes looking for reasons it might be *wrong*, or does it just
converge on the first source that agrees with the claim and call it done?

## Method

Both arms get the same tools: web search, page fetch, a source-credibility
lookup against a ~485-domain reference table, and a cheap model call that
judges whether a fetched page's text supports, refutes, or is mixed on the
claim. The baseline runs exactly one search and produces one verdict. The
agent runs a ReAct loop — reason, act, observe, repeat — up to a 15-step
safety cap, and its only way to stop is to call `submit_verdict`, which is
itself a guarded tool call, not a free action: it hard-rejects unless the
agent has already gathered at least one piece of evidence that disconfirms,
complicates, or is mixed on the claim. That guardrail exists specifically to
stop the agent from stopping at the first source that agrees with the claim.
It also turns out to be the main driver of the result below.

Every step — the model's reasoning, the tool call, the tool's result — is
logged to a structured trace. The trace is what gets scored, not just the
final verdict.

## Eval design

58 claims (19 true / 20 mixed / 19 false — a deliberately balanced split, not
just in-range), pulled from PolitiFact fact-checks and reviewed by hand, each
with the fact-checker's own citation trail as ground truth for what "the
agent found real evidence" means. A verdict counts as correct only if it
matches PolitiFact's bucket. Whether the agent *sought* disconfirming evidence
is tracked separately from whether it *found* evidence matching one of the
claim's real citation domains — the first is about behavior, the second is
about whether that behavior actually located something real.

One eval-specific guardrail: because these claims come from PolitiFact's own
fact-checks, a plain search for a claim's exact wording often surfaces
PolitiFact's own conclusion directly, letting the agent cite the answer key
instead of doing independent work. `tool_search` and `tool_fetch` hard-reject
domains tagged `fact-checker` in the credibility table, for eval runs only.
More on why that doesn't fully close the leak below.

## Results

| | Baseline | Agent (raw) | Agent (completed only) |
|---|---|---|---|
| Accuracy | 75.9% (44/58) | 72.4% (42/58) | **89.4%** (42/47) |
| Sought disconfirming evidence | 0% | 82.8% | — |
| Avg. steps | 2.0 (fixed) | 11.6 | — |
| Avg. tokens/claim | ~2,979 | ~133,371 | — |
| Approx. cost/claim | ~$0.009 | ~$0.30 | — |

The raw number makes the agent look like it lost. It didn't — it *finished*
less often. Every claim the agent actually completed an investigation for, it
judged better than the baseline did, by a wide margin. The honest framing
isn't "the agent is better" or "the agent is worse," it's: better judgment
when it finishes, and it doesn't reliably finish.

## Why investigations don't finish

Stuck rate — hitting the 15-step cap with no verdict — splits by ground truth
in a way that isn't noise: **true claims 26.3% (5/19), mixed 15.0% (3/20),
false 15.8% (3/19).** That skew is the guardrail doing exactly what it says,
applied to a case it can't handle: for a claim that's actually true, there
may be no legitimate disconfirming evidence in existence. The agent can't
satisfy a rule that requires finding something that isn't there — so it keeps
searching, keeps getting rejected, and burns its whole step budget.

One real trace makes this concrete. Claim `politifact-0004` — *"Florida is
the only state in the nation kicking children off their affordable health
coverage"* — is true. Step 1, a search immediately surfaces a PolitiFact
Florida piece republished under `wlrn.org` (an NPR affiliate's domain, not
tagged `fact-checker` in the credibility table), which the agent fetches and
treats as an independent source — a real instance of the fact-checker
exclusion's syndication gap discussed below. It goes on to independently
corroborate the claim via Georgetown's Center for Children & Families and a
Florida-focused health-policy nonprofit. At step 7 it tries fetching
`politifact.com` directly and gets `REJECTED: politifact.com is excluded from
this investigation. Use independent primary sources instead.` — the intended
guardrail working correctly. At step 12, with three independent corroborating
sources in hand, it tries to submit a `true` verdict and gets `REJECTED: You
must gather at least one piece of disconfirming evidence before submitting a
verdict.` Step 13's reasoning: *"I need to find explicit disconfirming
evidence. Let me search for Florida's defense or a conservative-leaning
source ch[allenging the claim]."* No such source exists to find, because the
claim is true. Two steps later, it hits the cap. Verdict: `null`. Scored as
wrong.

This is reported as a limitation of hard-enforced guardrails, not fixed and
rerun. A soft version — a strong prompt instruction instead of a hard
rejection — would very likely have let the agent submit `true` here with its
three corroborating sources. It would also plausibly let the agent stop at
the first agreeable source on claims where disconfirming evidence *does*
exist, which is the exact failure mode the guardrail was built to prevent.
That tradeoff wasn't run as a second experiment — a deliberate scope call for
a portfolio project, flagged here rather than left implicit.

**Failure attribution for the 5 completed-but-wrong agent verdicts:** 1
`reasoning_failure` (found the right evidence, still concluded wrong), 4
`search_failure` (never found a source matching the claim's real citation
domains). The baseline's 14 wrong claims are all `search_failure` by
construction — it never runs an evidence-gathering step at all, so it isn't a
finding about the baseline's reasoning, just the metric's mechanical floor
for an arm with nothing to attribute.

## Limitations

**The syndication leak.** Domain-blocking known fact-checker sites closes the
direct route to the answer key but not all of it, as the trace above shows
directly: PolitiFact content republished verbatim under a partner outlet's
own domain (`wlrn.org`) isn't tagged as a fact-checker source, so it passes
the filter untouched. This is a structural limit of using a public
fact-checking database as ground truth for a search-equipped agent, not a bug
with a fix — new syndication partners can't be enumerated in advance, and
chasing an ever-expanding blocklist wasn't worth doing for a fixed 58-claim
eval set.

**The credibility table's provenance.** The ~485-domain credibility/bias
table wasn't independently curated, despite that being the original Phase 1
plan specifically to avoid this. It was compiled from two publicly available
rating sources (Reality Team, Ad Fontes Media's free rankings) and parsed
into this project's schema with LLM help. The ratings reflect those two
organizations' methodologies, not independent editorial judgment — worth
knowing before treating any specific domain's rating as authoritative.

**A methodology note worth keeping, not just a bug fix.** During
development, one tool call returned a JSON boolean instead of the string
`"false"` for a verdict field — a type mismatch that would have silently
zeroed every accuracy score downstream if it had shipped. It was only caught
by checking real API output directly; the mocked test suite couldn't have
caught it, because the mock never had a reason to return anything but exactly
what the test expected. It's the concrete reason this project treated a real
end-to-end run against live APIs, not a green test suite, as the actual bar
for "done" at every phase — noted in the [README's development
section](README.md) accordingly.

## So, does autonomy earn its cost here?

Not a clean win. When the agent finishes an investigation, its judgment is
substantially better than a single search-and-answer baseline's — 89.4%
against 75.9% is a real gap, not noise. But "when it finishes" is doing a lot
of work in that sentence: a mechanistically understood 19% of the time, it
doesn't, for a reason specific to how one of its own guardrails interacts
with claims that are actually true. And the cost gap — roughly 34x in tokens,
unconditionally, whether the investigation completes or not — holds
regardless of how the accuracy question shakes out. The honest summary is
that autonomy buys better reasoning conditional on completing, at a real and
constant cost, with a specific, fixable-but-unfixed failure mode governing
whether it completes at all.

## What I'd do differently

The guardrail-starvation problem has an obvious next experiment: replace the
hard rejection in `submit_verdict` with a soft version — instruct the agent
strongly to seek disconfirming evidence, but let it submit anyway after a
certain number of failed attempts, with that fact logged and scored
separately. That would separate "the agent tried and genuinely couldn't find
disconfirming evidence" from "the agent gave up too early," which the current
binary reject/allow guardrail can't distinguish. It wasn't run here — the
15-step cap and hard guardrail were treated as fixed for this eval so the
58-claim run stayed a single, comparable experiment rather than turning into
a second study. That's the natural follow-up if this project continues past
the portfolio version.

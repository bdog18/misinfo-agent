"""Phase 4: single-shot RAG control arm."""
from misinfo_agent import tools, trace, agent


BASELINE_SYSTEM_PROMPT = (
    "You are a fact-checker. You get one shot: below are search results for a claim. "
    "Based only on these, decide true/false/mixed with a confidence and reasoning. "
    "You cannot search again, fetch pages, or ask for more information."
)


def run_baseline(claim: str, *, exclude_domains: set[str] | None = None) -> trace.Investigation:
    """Run the baseline agent."""
    # Create a new investigation
    investigation = trace.Investigation(
        claim=claim,
        arm="baseline",
        model=agent.ORCHESTRATOR_MODEL
    )

    # Step 1: Retrieve relevant information using the RAG tool
    search_observation = tools.tool_search(claim, exclude_domains=exclude_domains)
    investigation.add_step(
        thought="Single-shot baseline: one search, then a verdict", 
        action="tool_search",
        action_input={"query": claim},
        observation=str(search_observation),
        input_tokens=0,  # Placeholder for token count
        output_tokens=0  # Placeholder for token count
    )
    
    # Step 2: Submit the retrieved information for a verdict
    response = tools._get_anthropic_client().messages.create(
        model=agent.ORCHESTRATOR_MODEL,
        max_tokens=2048,
        system=BASELINE_SYSTEM_PROMPT,
        tools=[agent.SUBMIT_VERDICT_TOOL],
        tool_choice={"type": "tool", "name": "submit_verdict"},
        messages=[{"role": "user", "content": f"Claim: {claim}\n\nSearch results:\n{search_observation}"}],
    )
    verdict_input = response.content[0].input
    investigation.verdict = str(verdict_input["verdict"]).lower()
    investigation.confidence = verdict_input["confidence"]
    investigation.reasoning = verdict_input["reasoning"]
    investigation.add_step(
        thought="Submitting search results for a verdict",
        action="submit_verdict",
        action_input=verdict_input,
        observation=f"VERDICT SUBMITTED: {str(verdict_input['verdict']).lower()}",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens
    )
    investigation.stop_reason = "verdict_submitted"
    
    return investigation
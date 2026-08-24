"""Phase 3: the ReAct investigation loop."""

from misinfo_agent import tools, trace


AGENT_TOOLS = [
    {
        "name": "tool_fetch",
        "description": (
            "Fetch and extract the readable text of a single url. Use this "
            "after tool_search to read a source in full before judging it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch and extract text from."
                }
            },
            "required": ["url"]
        }  
    },
    {
        "name": "tool_search",
        "description": (
            "Search the web for a given query and return a list of relevant results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to use."
                },
                "max_results": {
                    "type": "integer",
                    "description": "The maximum number of search results to return."
                }
            },
            "required": ["query"]
        }  
    },
    {
        "name": "tool_assess_source",
        "description": (
            "Assess the credibility and reliability of a given source."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch and extract text from."
                }
            },
            "required": ["url"]
        }  
    },
    {
        "name": "tool_compare_claim_to_text",
        "description": (
            "Compare a claim to the text of a source to determine if the claim is supported by the text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The claim to compare to the text of the source."
                },
                "text": {
                    "type": "string",
                    "description": "The text of the source to compare the claim to."
                },
                "url": {
                    "type": "string",
                    "description": "The URL to fetch and extract text from."
                }
            },
            "required": ["claim", "text", "url"]
        }  
    },
    {
        "name": "submit_verdict",
        "description": (
            "Submit a final verdict based on the investigation taking into account disconfirming evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "enum": ["true", "false", "mixed"],
                    "description": "The final verdict based on the investigation."
                },
                "confidence": {
                    "type": "number",
                    "description": "The confidence level of the verdict."
                },
                "reasoning": {
                    "type": "string",
                    "description": "The reasoning behind the verdict."
                }
            },
            "required": ["verdict", "confidence", "reasoning"]
        }
    },
]


TOOL_DISPATCH = {
    "tool_fetch": tools.tool_fetch,
    "tool_search": tools.tool_search,
    "tool_assess_source": tools.tool_assess_source,
    "tool_compare_claim_to_text": tools.tool_compare_claim_to_text
}


MAX_HITS_PER_DOMAIN = 3


SYSTEM_PROMPT = (
    "You are an investigative fact-checker. You will be given a claim, and your "
    "job is to reach a verdict on whether it is true, false, or mixed by actively "
    "gathering and weighing real evidence — not by relying on what you already "
    "believe about it.\n\n"
    
    "You work step by step: at each turn, decide on ONE next action (a single "
    "tool call), take it, and read the observation before deciding what to do "
    "next. Do not assert evidence you have not actually retrieved, and make sure"
    "before EVERY action, ALWAYS briefly state what you're doing and why. "
    "Always assess a source before relying on its text. \n\n"
    
    "Your tools:\n"
    "- tool_search: search the web for a query.\n"
    "- tool_fetch: fetch and read the full text of a specific url.\n"
    "- tool_assess_source: look up a url's credibility and political bias "
    "rating. Use this on the sources you rely on, and weigh low-credibility or "
    "heavily biased sources accordingly.\n"
    "- tool_compare_claim_to_text: check whether a fetched text supports, "
    "refutes, or is irrelevant to the claim.\n"
    "- submit_verdict: your final action. Call it with a verdict (\"true\", "
    "\"false\", or \"mixed\"), a confidence (0.0-1.0), and your reasoning.\n\n"
    
    "Three rules are enforced by the system itself, not just good practice:\n"
    f"1. You may fetch a given domain at most {MAX_HITS_PER_DOMAIN} times. "
    "Trying again beyond that will be rejected — diversify your sources instead "
    "of re-reading the same one.\n"
    "2. submit_verdict will be REJECTED unless you have already gathered at "
    "least one piece of evidence that disconfirms, complicates, or is mixed on "
    "the claim — not only evidence that agrees with it. If everything you've "
    "found so far agrees with the claim, you must actively look for a source "
    "that might disagree before you are allowed to submit.\n"
    "3. tool_compare_claim_to_text will be REJECTED for a url you have not "
    "already called tool_assess_source on. Always assess a source before "
    "comparing its text to the claim.\n\n"
    
    "Reach a verdict efficiently once you have genuinely weighed both sides — "
    "do not pad the investigation with redundant searches once the evidence is "
    "sufficient."
)

ORCHESTRATOR_MODEL = "claude-sonnet-5"


def _domain_hit_count(investigation: trace.Investigation, domain: str) -> int:
    return sum(1 for step in investigation.steps if step.action == "tool_fetch" and domain == tools._extract_domain(step.action_input.get("url", "")))


def _sought_disconfirming_evidence(investigation: trace.Investigation) -> bool:
    return any(evidence.stance == "refutes" or evidence.stance == "mixed" for evidence in investigation.evidence)


def run_investigation(claim: str, max_steps: int = 15) -> trace.Investigation:
    """Run a ReAct investigation loop on a claim."""
    investigation = trace.Investigation(claim=claim, arm="agent", model=ORCHESTRATOR_MODEL)
    messages = [
        {"role": "user", "content": claim}
    ]
    source_ratings: dict[str, tools.SourceAssessment] = {}

    while len(investigation.steps) < max_steps:
        response = tools._get_anthropic_client().messages.create(
            model=ORCHESTRATOR_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=AGENT_TOOLS,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})
        
        thought = "".join(b.text for b in response.content if b.type == "text")
        tool_block = next(b for b in response.content if b.type == "tool_use")
        action, action_input = tool_block.name, tool_block.input

        if action == "submit_verdict":
            missing = [f for f in ("verdict", "confidence", "reasoning") if f not in action_input]
            if missing:
                observation = (
                    f"REJECTED: Your submit_verdict call was missing required field(s) "
                    f"{missing}. Please call submit_verdict again with all of verdict, "
                    f"confidence, and reasoning."
                )
            elif not _sought_disconfirming_evidence(investigation):
                observation = "REJECTED: You must gather at least one piece of disconfirming evidence before submitting a verdict."
            else:
                investigation.verdict = action_input["verdict"]
                investigation.confidence = action_input["confidence"]
                investigation.reasoning = action_input["reasoning"]
                investigation.stop_reason = "verdict_submitted"
                observation = "VERDICT SUBMITTED"
        elif action == "tool_fetch":
            url = action_input.get("url", "")
            domain = tools._extract_domain(url)
            if _domain_hit_count(investigation, domain) >= MAX_HITS_PER_DOMAIN:
                observation = f"REJECTED: You have already fetched {MAX_HITS_PER_DOMAIN} urls from {domain}. Please diversify your sources."
            else:
                result = TOOL_DISPATCH["tool_fetch"](**action_input)
                observation = str(result)
        elif action == "tool_assess_source":
            result = TOOL_DISPATCH["tool_assess_source"](**action_input)
            source_ratings[action_input["url"]] = result
            observation = str(result)
        elif action == "tool_compare_claim_to_text":
            url = action_input["url"]
            if url not in source_ratings:
                observation = (
                    f"REJECTED: You must call tool_assess_source on {url} before "
                    "comparing its text to the claim. Assess it first, then retry "
                    "this comparison."
                )
            else:
                dispatch_args = {k: v for k, v in action_input.items() if k != "url"}
                result = TOOL_DISPATCH["tool_compare_claim_to_text"](**dispatch_args)
                observation = str(result)

                rating = source_ratings[url]
                investigation.add_evidence(trace.Evidence(
                    url=url,
                    stance=result.stance,
                    confidence=result.confidence,
                    quote=result.quote,
                    source_credibility=rating.credibility,
                    source_bias=rating.bias,
                    step_number=len(investigation.steps) + 1,
                ))
        else:
            result = TOOL_DISPATCH[action](**action_input)
            observation = str(result)
            
            
        # Record the step
        investigation.add_step(
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": observation,
            }]
        })

        if investigation.stop_reason == "verdict_submitted":
            break

    if len(investigation.steps) >= max_steps:
        investigation.stop_reason = "max_steps_reached"

    return investigation


if __name__ == "__main__":
    import sys

    claim_text = sys.argv[1] if len(sys.argv) > 1 else "The measles vaccine causes autism."
    investigation = run_investigation(claim_text)

    for step in investigation.steps:
        print(f"\n--- Step {step.step_number}: {step.action} ---")
        print(f"Thought: {step.thought}")
        print(f"Input: {step.action_input}")
        print(f"Observation: {step.observation}")

    print(f"\n=== Verdict: {investigation.verdict} (confidence={investigation.confidence}) ===")
    print(f"Reasoning: {investigation.reasoning}")
    print(f"Stop reason: {investigation.stop_reason}")
    print(f"Steps taken: {len(investigation.steps)}")
    print(f"Tokens: {investigation.total_input_tokens} in / {investigation.total_output_tokens} out")
    print(f"\nEvidence gathered ({len(investigation.evidence)}):")
    for ev in investigation.evidence:
        print(f"  [{ev.stance}] {ev.url} (source credibility={ev.source_credibility}, bias={ev.source_bias})")
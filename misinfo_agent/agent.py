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
                }
            },
            "required": ["claim", "text"]
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
    "next. Do not assert evidence you have not actually retrieved.\n\n"
    
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
    
    "Two rules are enforced by the system itself, not just good practice:\n"
    f"1. You may fetch a given domain at most {MAX_HITS_PER_DOMAIN} times. "
    "Trying again beyond that will be rejected — diversify your sources instead "
    "of re-reading the same one.\n"
    "2. submit_verdict will be REJECTED unless you have already gathered at "
    "least one piece of evidence that disconfirms, complicates, or is mixed on "
    "the claim — not only evidence that agrees with it. If everything you've "
    "found so far agrees with the claim, you must actively look for a source "
    "that might disagree before you are allowed to submit.\n\n"
    
    "Reach a verdict efficiently once you have genuinely weighed both sides — "
    "do not pad the investigation with redundant searches once the evidence is "
    "sufficient."
)

def _domain_hit_count(investigation: trace.Investigation, domain: str) -> int:
    return sum(1 for step in investigation.steps if step.action == "tool_fetch" and domain == tools._extract_domain(step.action_input.get("url", "")))

def _sought_disconfirming_evidence(investigation: trace.Investigation) -> bool:
    return any(evidence.stance == "refutes" or evidence.stance == "mixed" for evidence in investigation.evidence)
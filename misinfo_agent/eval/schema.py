"""Ground-truth claim schema for the eval set (misinfo_agent/eval/claims.jsonl).

Verdict bucket mapping (documented here because it's a methodology choice,
not an implementation detail — it determines what "correct" means when we
score agent output against ground truth):

    true   <- PolitiFact "True", "Mostly True"
    mixed  <- PolitiFact "Half True"
    false  <- PolitiFact "Mostly False", "False", "Pants on Fire"

"Mostly False" is bucketed with "false" rather than "mixed": PolitiFact's
own definition of Mostly False is "the statement contains an element of
truth but ignores critical facts that would give a different impression" —
closer to false-with-a-kernel-of-truth than genuinely balanced. Only "Half
True" (roughly matched pros/cons) lands in "mixed". This is a judgment
call; `verdict_detail` preserves the original label so the mapping can be
revisited without re-collecting data.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

Verdict = Literal["true", "false", "mixed"]

POLITIFACT_VERDICT_MAP: dict[str, Verdict] = {
    "true": "true",
    "mostly-true": "true",
    "half-true": "mixed",
    "mostly-false": "false",
    "false": "false",
    "pants-fire": "false",
}


class Citation(BaseModel):
    publisher: str
    title: str
    url: str
    cited_date: str | None = None


class Claim(BaseModel):
    id: str = Field(description="Stable id, e.g. 'politifact-0001'")
    source: str = Field(description="Fact-checking org this claim/verdict came from")
    claim: str = Field(min_length=10, description="The claim text as stated by the claimant")
    claimant: str | None = None
    date_claimed: date | None = None
    date_checked: date | None = None
    verdict: Verdict = Field(description="Bucketed ground-truth label used for scoring")
    verdict_detail: str = Field(description="Original fact-checker rating, e.g. 'Mostly False'")
    categories: list[str] = Field(default_factory=list)
    fact_check_url: str = Field(description="URL of the fact-check article itself")
    citations: list[Citation] = Field(
        default_factory=list,
        description="The citation trail: sources the fact-checker relied on",
    )
    reviewed: bool = Field(
        default=False,
        description="True once a human has confirmed claim text, verdict, and citations",
    )
    notes: str | None = None

    @field_validator("fact_check_url")
    @classmethod
    def _valid_url(cls, v: str) -> str:
        HttpUrl(v)
        return v

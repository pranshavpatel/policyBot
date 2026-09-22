"""Groundedness guardrail: catch answers that state a number/date not
actually present in the retrieved context (the most common, most checkable
form of RAG hallucination for a policy bot — "12 weeks of parental leave"
when the context says 6 is a very different, very costly wrong answer).

This is a cheap lexical check, not an LLM-judge: it costs zero extra tokens
and zero extra latency, which matters because it runs on every answer.
An LLM-as-judge check would catch more (e.g. a claim that inverts a
qualitative rule with no numbers in it), but at 1 extra call per answer;
left as a documented option (`llm_judge_groundedness`) rather than the
default path.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List

from config import GROUNDEDNESS_MIN_COVERAGE

# Matches things like: 15, 15.5, 12%, $50, 90 days, Jan 1, 07-04, 2025-09-18
_NUMERIC_RE = re.compile(
    r"""
    \$?\d[\d,]*(?:\.\d+)?%?      # 15 | 15.5 | $50 | 12%
    """,
    re.VERBOSE,
)


def _extract_claims(text: str) -> List[str]:
    """Pull out numeric/date tokens an answer makes factual claims with."""
    return [m.group(0) for m in _NUMERIC_RE.finditer(text)]


def _normalize(token: str) -> str:
    return token.strip().rstrip(".,;:").lower()


@dataclass
class GroundednessResult:
    grounded: bool
    coverage: float                # fraction of claims found in context (1.0 if no claims)
    unsupported_claims: List[str] = field(default_factory=list)


def check_groundedness(answer: str, context: str, min_coverage: float = GROUNDEDNESS_MIN_COVERAGE) -> GroundednessResult:
    claims = _extract_claims(answer)
    if not claims:
        return GroundednessResult(grounded=True, coverage=1.0)

    context_norm = _normalize(context)
    unsupported = [c for c in claims if _normalize(c) not in context_norm]
    coverage = 1.0 - (len(unsupported) / len(claims))
    return GroundednessResult(
        grounded=coverage >= min_coverage,
        coverage=round(coverage, 3),
        unsupported_claims=unsupported,
    )


UNGROUNDED_FALLBACK = (
    "I found a possible answer in the policy docs but couldn't verify every "
    "number in it against the source text, so I don't want to state it with "
    "confidence. Please check the policy document directly, or rephrase your "
    "question."
)

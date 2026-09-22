"""Regression tests for the ReAct-loop groundedness safety net
(agent/react_agent.py::_ground_final_answer), added after a live eval run
caught a real hallucination: the planner's "final" action answered a VPN
support question with a fabricated email/extension instead of routing
through — or being checked against — retrieved policy context."""
from agent.react_agent import _ground_final_answer
from rag.groundedness import UNGROUNDED_FALLBACK


def test_grounded_claim_passes_through():
    ctx = ["Employees receive 10 paid sick days annually."]
    answer = "You get 10 paid sick days per year."
    assert _ground_final_answer(answer, ctx) == answer


def test_unsupported_claim_with_context_is_blocked():
    ctx = ["Employees receive 10 paid sick days annually."]
    answer = "You get 25 paid sick days per year."
    assert _ground_final_answer(answer, ctx) == UNGROUNDED_FALLBACK


def test_factual_claim_with_no_context_at_all_is_blocked():
    # The exact failure mode this was written for: the planner answers a
    # policy question directly (no rag_answer tool call happened this run,
    # so rag_context_parts is empty) and invents a specific detail.
    answer = "Contact the IT Help Desk at helpdesk@company.com or ext. 1234."
    assert _ground_final_answer(answer, []) == UNGROUNDED_FALLBACK


def test_greeting_with_no_context_passes():
    # No numeric/date claims -> nothing to be unsupported -> should not be
    # blocked just for having empty context.
    answer = "Hi! Ask me about PTO/leave policies or request/approve leave."
    assert _ground_final_answer(answer, []) == answer


def test_multiple_rag_answer_calls_are_pooled():
    ctx = ["Sick leave: 10 days annually.", "PTO Year 1: 15 days."]
    answer = "You get 15 days of PTO in Year 1."
    assert _ground_final_answer(answer, ctx) == answer

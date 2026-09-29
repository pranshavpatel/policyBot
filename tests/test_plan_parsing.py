"""Unit tests for agent/react_agent.py's plan-JSON recovery.

Regression tests for a real failure in a live qwen3:32b eval run: one
question got "Sorry, I couldn't parse a plan" back instead of an answer.
Some models (Qwen's "thinking" mode is the documented case) emit a
<think>...</think> reasoning block before the JSON even when explicitly
told to output JSON only — _extract_json_object recovers from that."""
import json
from agent.react_agent import _extract_json_object, plan_step
import agent.react_agent as react_agent_module


def test_extracts_json_after_think_block():
    raw = '<think>I should use the rag_answer tool for this.</think>\n{"action":"tool","name":"rag_answer","args":{"query":"MFA"}}'
    extracted = _extract_json_object(raw)
    assert json.loads(extracted) == {"action": "tool", "name": "rag_answer", "args": {"query": "MFA"}}


def test_extracts_json_with_no_think_block():
    raw = '{"action":"final","answer":"Hi!"}'
    extracted = _extract_json_object(raw)
    assert json.loads(extracted) == {"action": "final", "answer": "Hi!"}


def test_returns_none_when_no_json_object_present():
    assert _extract_json_object("<think>just rambling, no JSON at all</think>") is None


def test_handles_nested_braces_in_args():
    raw = '<think>...</think>{"action":"tool","name":"http_post","args":{"json":{"nested":"value"}}}'
    extracted = _extract_json_object(raw)
    assert json.loads(extracted)["args"]["json"]["nested"] == "value"


class _FakeMessage:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {}


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, prompt):
        return _FakeMessage(self._content)


def test_plan_step_recovers_from_think_block(monkeypatch):
    # End-to-end through plan_step() itself, not just the helper — proves
    # the recovery is actually wired in, not just unit-tested in isolation.
    fake_content = '<think>reasoning...</think>\n{"action":"final","answer":"ok"}'
    monkeypatch.setattr(react_agent_module, "_llm", lambda: _FakeLLM(fake_content))
    plan = plan_step("hello")
    assert plan == {"action": "final", "answer": "ok"}


def test_plan_step_still_fails_gracefully_on_genuinely_unparseable_output(monkeypatch):
    monkeypatch.setattr(react_agent_module, "_llm", lambda: _FakeLLM("not json at all, no braces"))
    plan = plan_step("hello")
    assert plan["action"] == "final"
    assert "couldn't parse" in plan["answer"].lower()

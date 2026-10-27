"""Regression tests for agent/react_agent.py::run_agent's loop control flow
— mocked planner and tools (no real LLM), no server needed. Covers a real
bug found live: "Request PTO from 2025-10-02 to 2025-10-04" against the
deployed bot created 3 duplicate leave requests (the planner kept
re-issuing the same successful create_leave_request call) and then had its
own confirmation blocked by the groundedness guardrail (the tool's own
echoed-back dates weren't in the grounding pool at all)."""
import agent.react_agent as react_agent_module
from agent.react_agent import run_agent


def _fake_tools(create_leave_calls=None):
    def _create_leave_request(args):
        if create_leave_calls is not None:
            create_leave_calls.append(args)
        return {"id": "req-1", "status": "submitted", "start_date": args.get("start_date"), "end_date": args.get("end_date")}

    return {"create_leave_request": {"fn": _create_leave_request}}


def test_successful_write_tool_terminates_loop_no_duplicate_call(monkeypatch):
    calls = []
    plan = {"action": "tool", "name": "create_leave_request",
            "args": {"start_date": "2025-10-02", "end_date": "2025-10-04"}}
    monkeypatch.setattr(react_agent_module, "plan_step", lambda context: plan)
    monkeypatch.setattr(react_agent_module, "TOOLS", _fake_tools(create_leave_calls=calls))
    monkeypatch.setattr(
        react_agent_module, "synthesize",
        lambda user_msg, tool_name, tool_output: f"Request submitted for {tool_output['start_date']} to {tool_output['end_date']}.",
    )

    result = run_agent(
        "Request PTO from 2025-10-02 to 2025-10-04",
        actor={"username": "alice", "role": "employee"},
        max_steps=3,
    )

    assert result["steps"] == 1
    assert len(calls) == 1  # not re-issued across steps
    assert "2025-10-02" in result["answer"]
    assert "couldn't verify" not in result["answer"]  # not blocked by groundedness

"""Unit tests for agent/graph.py's routing logic — mocked planner and
tools (no real LLM/server), so these run everywhere the rest of the suite
does. Real end-to-end verification (deterministic single-step routing cut
a controlled question from 70.2s/2 steps to 29.7s/1 step against
qwen3:8b, and conversation memory across two real turns) is documented in
the README rather than reproduced here, since CI has no local model to
call out to."""
import agent.graph as graph_module
from agent.graph import build_graph
from langgraph.checkpoint.memory import MemorySaver


def _fake_tools(rag_answer_result=None, rag_answer_error=False, create_leave_calls=None):
    def _rag_answer(args):
        if rag_answer_error:
            return {"error": "boom"}
        return rag_answer_result or {"answer": "15 days", "citations": [], "context": "Year 1: 15 days", "grounded": True}

    def _create_leave_request(args):
        if create_leave_calls is not None:
            create_leave_calls.append(args)
        return {"id": "req-1", "status": "submitted", "start_date": args.get("start_date"), "end_date": args.get("end_date")}

    return {
        "rag_answer": {"fn": _rag_answer},
        "check_holiday": {"fn": lambda args: {"is_holiday": False, "name": None, "date": args.get("date_str")}},
        "create_leave_request": {"fn": _create_leave_request},
    }


def _make_graph(monkeypatch, plans, tools=None):
    """plans: a list of dicts returned by successive plan_step() calls
    (one per planner invocation this test run makes)."""
    calls = {"i": 0}

    def _fake_plan_step(context):
        i = calls["i"]
        calls["i"] += 1
        return plans[min(i, len(plans) - 1)]

    monkeypatch.setattr(graph_module, "plan_step", _fake_plan_step)
    monkeypatch.setattr(graph_module, "synthesize", lambda user_msg, tool_name, tool_output: tool_output.get("answer", ""))
    monkeypatch.setattr(graph_module, "TOOLS", tools or _fake_tools())
    return build_graph(checkpointer=MemorySaver())


def test_direct_answer_tool_routes_straight_to_finalize_no_second_plan_call(monkeypatch):
    # The actual fix: after a successful rag_answer, no second planner call
    # should happen at all — this is what cut a real question from 2 LLM
    # calls to 1 against a real model (see README).
    plans = [{"action": "tool", "name": "rag_answer", "args": {"query": "PTO Year 1"}}]
    graph = _make_graph(monkeypatch, plans)
    result = graph.invoke(
        {"user_msg": "How many PTO days in Year 1?", "actor": {"username": "alice", "role": "employee"},
         "max_steps": 3, "messages": [], "steps": 0, "trace": [], "rag_context_parts": [], "final_answer": None},
        config={"configurable": {"thread_id": "t1"}},
    )
    assert result["steps"] == 1
    assert "15 days" in result["final_answer"]
    # Exactly one planner call was ever available (plans has length 1) —
    # if the graph had looped back to plan a second time it would have
    # reused plans[0] again (min() clamp) rather than erroring, so the
    # real guarantee here is steps == 1 above, not a call-count assertion.


def test_final_action_with_no_tool_skips_execute_tool_entirely(monkeypatch):
    plans = [{"action": "final", "answer": "Hi there!"}]
    graph = _make_graph(monkeypatch, plans)
    result = graph.invoke(
        {"user_msg": "hello", "actor": None, "max_steps": 3, "messages": [], "steps": 0,
         "trace": [], "rag_context_parts": [], "final_answer": None},
        config={"configurable": {"thread_id": "t2"}},
    )
    assert result["final_answer"] == "Hi there!"
    assert result["steps"] == 1
    assert not any("tool_call" in t for t in result["trace"])


def test_tool_error_does_not_route_to_finalize_and_retries_plan(monkeypatch):
    plans = [
        {"action": "tool", "name": "rag_answer", "args": {"query": "x"}},
        {"action": "final", "answer": "recovered"},
    ]
    graph = _make_graph(monkeypatch, plans, tools=_fake_tools(rag_answer_error=True))
    result = graph.invoke(
        {"user_msg": "x", "actor": {"username": "alice", "role": "employee"}, "max_steps": 3,
         "messages": [], "steps": 0, "trace": [], "rag_context_parts": [], "final_answer": None},
        config={"configurable": {"thread_id": "t3"}},
    )
    # an error result should NOT be treated as a direct-answer success —
    # the loop should return to plan rather than finalize on a failure
    assert result["steps"] == 2
    assert result["final_answer"] == "recovered"


def test_unknown_tool_name_finalizes_with_error_message(monkeypatch):
    plans = [{"action": "tool", "name": "not_a_real_tool", "args": {}}]
    graph = _make_graph(monkeypatch, plans)
    result = graph.invoke(
        {"user_msg": "x", "actor": None, "max_steps": 3, "messages": [], "steps": 0,
         "trace": [], "rag_context_parts": [], "final_answer": None},
        config={"configurable": {"thread_id": "t4"}},
    )
    assert "not_a_real_tool" in result["final_answer"]


def test_successful_write_tool_routes_straight_to_finalize_no_duplicate_call(monkeypatch):
    # Regression test for a real bug found live: after a successful
    # create_leave_request, the graph looped back to plan and re-issued
    # the same write, creating duplicate leave requests in production.
    # Any successful tool call — not just an allowlisted subset of
    # read-only ones — must be terminal.
    calls = []
    plans = [{"action": "tool", "name": "create_leave_request",
              "args": {"start_date": "2025-10-02", "end_date": "2025-10-04"}}]
    graph = _make_graph(monkeypatch, plans, tools=_fake_tools(create_leave_calls=calls))
    monkeypatch.setattr(
        graph_module, "synthesize",
        lambda user_msg, tool_name, tool_output: f"Request submitted for {tool_output['start_date']} to {tool_output['end_date']}.",
    )
    result = graph.invoke(
        {"user_msg": "Request PTO from 2025-10-02 to 2025-10-04",
         "actor": {"username": "alice", "role": "employee"}, "max_steps": 3,
         "messages": [], "steps": 0, "trace": [], "rag_context_parts": [], "final_answer": None},
        config={"configurable": {"thread_id": "t5"}},
    )
    assert result["steps"] == 1
    assert len(calls) == 1  # not re-issued
    assert "2025-10-02" in result["final_answer"]
    assert "couldn't verify" not in result["final_answer"]  # not blocked by groundedness either


def test_conversation_memory_carries_rag_context_across_turns(monkeypatch):
    # Regression test for a real issue found testing this against a live
    # model: a follow-up answered from conversation memory (no new
    # rag_answer call) still needs the PRIOR turn's retrieved context to
    # pass groundedness — otherwise a correct memory-based numeric answer
    # gets blocked with had_context=False every time.
    from agent.graph import run_agent_graph

    plans_turn1 = [{"action": "tool", "name": "rag_answer", "args": {"query": "wellness stipend"}}]
    monkeypatch.setattr(graph_module, "TOOLS", _fake_tools(
        rag_answer_result={"answer": "$50/month", "citations": [], "context": "Wellness stipend: $50/month", "grounded": True}))
    calls = {"i": 0}

    def _fake_plan_step_turn1(context):
        return plans_turn1[0]

    monkeypatch.setattr(graph_module, "plan_step", _fake_plan_step_turn1)
    monkeypatch.setattr(graph_module, "synthesize", lambda user_msg, tool_name, tool_output: tool_output.get("answer", ""))
    # Force a fresh graph singleton for this test's checkpointer
    monkeypatch.setattr(graph_module, "_compiled_graph", None)

    r1 = run_agent_graph("How much is the wellness stipend?", actor={"username": "carol", "role": "employee"},
                          thread_id="memtest")
    assert "$50" in r1["answer"]

    # Turn 2: planner answers directly from memory, no tool call — must
    # still be grounded against turn 1's carried-forward context.
    monkeypatch.setattr(graph_module, "plan_step", lambda context: {"action": "final", "answer": "Yes, $50/month."})
    r2 = run_agent_graph("Is that the same one?", actor={"username": "carol", "role": "employee"},
                          thread_id="memtest")
    assert "$50" in r2["answer"]
    assert "couldn't verify" not in r2["answer"]

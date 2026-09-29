"""LangGraph-based agent engine — an alternative to the hand-rolled loop
in agent/react_agent.py, toggled via config.AGENT_ENGINE (same "config
flip, not a rewrite" pattern as RETRIEVAL_MODE/LLM_PROVIDER).

Two things this fixes that the loop couldn't, by construction rather than
by asking the model nicely:

- The redundant-re-call latency bug (README: "the ReAct loop re-planning
  2-3 steps per question when one rag_answer call would do"). After a
  successful rag_answer (or another direct-answer tool), the graph's edge
  routes straight to `finalize` — a deterministic graph edge, not an
  LLM's free-form judgment call about whether to stop. There's no prompt
  wording for the model to ignore, because the model is never asked.

- Statefulness. A checkpointer gives each conversation thread (keyed by
  the authenticated user by default) persistent memory across requests,
  so a follow-up question can reference the prior turn — the hand-rolled
  loop is single-turn only, by design.

Reuses agent/react_agent.py's plan_step / synthesize / _ground_final_answer
/ PLANNER_PROMPT rather than reimplementing them: only the control flow
(when to call the LLM again vs. stop) changes here, not the LLM-calling
logic itself.
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .tools import TOOLS
from .react_agent import plan_step, synthesize, _ground_final_answer, ACTOR_SCOPED_TOOLS
from observability import get_logger, log_event

log = get_logger("agent.graph")

# Tools whose successful result is, by construction, a complete answer —
# routing straight to finalize instead of back through the planner is
# what actually eliminates the redundant-re-call bug for these.
DIRECT_ANSWER_TOOLS = {"rag_answer", "doc_search", "check_holiday"}


class AgentState(TypedDict, total=False):
    user_msg: str
    actor: Optional[Dict[str, str]]
    messages: List[Dict[str, str]]           # prior turns, for conversation memory
    plan: Dict[str, Any]
    tool_name: Optional[str]
    tool_output: Optional[Dict[str, Any]]
    rag_context_parts: List[str]
    steps: int
    max_steps: int
    final_answer: Optional[str]
    trace: List[Dict[str, Any]]


def _render_history(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _plan_node(state: AgentState) -> AgentState:
    history = _render_history(state.get("messages", []))
    context = f"{history}\n{state['user_msg']}".strip() if history else state["user_msg"]
    plan = plan_step(context)
    state["plan"] = plan
    state["steps"] = state.get("steps", 0) + 1
    state.setdefault("trace", []).append({"step": state["steps"], "plan": plan})
    return state


def _execute_tool_node(state: AgentState) -> AgentState:
    plan = state["plan"]
    name = plan.get("name")
    args = dict(plan.get("args", {}) or {})
    tool = TOOLS.get(name)
    if not tool:
        state["final_answer"] = f"Unknown tool '{name}'."
        state["tool_output"] = None
        return state

    if name in ACTOR_SCOPED_TOOLS:
        args["_actor"] = state.get("actor")

    t0 = time.perf_counter()
    try:
        obs = tool["fn"](args)
    except Exception as e:
        obs = {"error": str(e)}
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    log_event(log, "tool_call", name=name, latency_ms=latency_ms,
              ok=("error" not in obs) if isinstance(obs, dict) else True)

    state.setdefault("trace", []).append(
        {"step": state["steps"], "tool_call": {"name": name}, "observation": obs, "latency_ms": latency_ms}
    )
    state["tool_name"] = name
    state["tool_output"] = obs
    if name == "rag_answer" and isinstance(obs, dict) and obs.get("context"):
        state.setdefault("rag_context_parts", []).append(obs["context"])
    return state


def _finalize_node(state: AgentState) -> AgentState:
    if state.get("final_answer"):
        pass  # already set by an earlier node (e.g. unknown-tool error)
    elif state["plan"].get("action") == "final":
        state["final_answer"] = _ground_final_answer(
            state["plan"].get("answer", ""), state.get("rag_context_parts", []))
    elif state.get("tool_output") is not None:
        synthesized = synthesize(state["user_msg"], state["tool_name"], state["tool_output"])
        state["final_answer"] = _ground_final_answer(synthesized, state.get("rag_context_parts", []))
    else:
        state["final_answer"] = "I couldn't decide on a next action."
    return state


def _route_after_plan(state: AgentState) -> str:
    # Route to execute_tool for ANY "tool" action, even an unknown tool
    # name — _execute_tool_node already handles that case and sets a
    # proper "Unknown tool 'x'" error message. Checking `name in TOOLS`
    # here too meant an unknown tool silently skipped straight to
    # finalize with no error context at all, producing the generic
    # "I couldn't decide on a next action." instead — caught by
    # tests/test_agent_graph.py::test_unknown_tool_name_finalizes_with_error_message.
    if state["plan"].get("action") == "tool":
        return "execute_tool"
    return "finalize"


def _route_after_tool(state: AgentState) -> str:
    obs = state.get("tool_output")
    ok = isinstance(obs, dict) and "error" not in obs
    if state.get("tool_name") in DIRECT_ANSWER_TOOLS and ok:
        return "finalize"  # the deterministic fix — no LLM re-judgment needed
    if state["steps"] >= state.get("max_steps", 3):
        return "finalize"
    return "plan"


def build_graph(checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("plan", _plan_node)
    graph.add_node("execute_tool", _execute_tool_node)
    graph.add_node("finalize", _finalize_node)
    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", _route_after_plan, {"execute_tool": "execute_tool", "finalize": "finalize"})
    graph.add_conditional_edges("execute_tool", _route_after_tool, {"finalize": "finalize", "plan": "plan"})
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


_compiled_graph = None


def get_graph():
    """Lazily-built module-level singleton, checkpointed with an in-memory
    saver — conversation threads live for the process's lifetime, which is
    the same durability model the rest of this app's per-process state
    (the hand-rolled loop has none at all) already has. Swap in a
    persistent checkpointer (langgraph.checkpoint.sqlite's SqliteSaver, the
    same storage engine leave_requests/audit_log already use) if
    conversations need to survive a restart — not done here to keep this
    change scoped to the control-flow fix and the memory it enables, not a
    new persistence layer."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


_MAX_CARRIED_CONTEXT_CHUNKS = 20  # cap unbounded growth over a long conversation


def run_agent_graph(user_msg: str, actor: Optional[Dict[str, str]] = None, max_steps: int = 3,
                     thread_id: Optional[str] = None) -> Dict[str, Any]:
    """Drop-in replacement for agent.react_agent.run_agent's return shape,
    so api/app.py doesn't need to change regardless of which AGENT_ENGINE
    is configured. thread_id defaults to the actor's username — one
    persistent conversation per signed-in user."""
    graph = get_graph()
    thread_id = thread_id or (actor or {}).get("username") or "anonymous"
    graph_config = {"configurable": {"thread_id": thread_id}}

    prior_state = graph.get_state(graph_config)
    prior_values = prior_state.values or {}
    messages = list(prior_values.get("messages", []))
    # Carried across turns, not just accumulated within one: a follow-up
    # that answers from conversation memory rather than a fresh rag_answer
    # call still needs *some* context to ground its claims against — found
    # live, testing this against a real model: "Is that the same stipend?"
    # got blocked as ungrounded even though the number was correct, because
    # the turn that actually retrieved it was the previous one. Without
    # this, every memory-based follow-up with a number in it would be
    # blocked by construction, not just when it should be.
    prior_context = list(prior_values.get("rag_context_parts", []))

    run_t0 = time.perf_counter()
    result = graph.invoke(
        {
            "user_msg": user_msg, "actor": actor, "max_steps": max_steps,
            "messages": messages, "steps": 0, "trace": [], "rag_context_parts": list(prior_context),
            "final_answer": None,
        },
        config=graph_config,
    )
    total_ms = round((time.perf_counter() - run_t0) * 1000, 1)

    updated_messages = messages + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": result["final_answer"]},
    ]
    updated_context = result.get("rag_context_parts", prior_context)[-_MAX_CARRIED_CONTEXT_CHUNKS:]
    graph.update_state(graph_config, {"messages": updated_messages, "rag_context_parts": updated_context})

    log_event(log, "agent_run", actor=(actor or {}).get("username"), steps=result["steps"],
              total_latency_ms=total_ms, outcome="graph", thread_id=thread_id)

    return {"type": "final", "answer": result["final_answer"], "steps": result["steps"],
            "trace": result.get("trace", [])}

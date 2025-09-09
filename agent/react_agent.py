# agent/react_agent.py
import json
from typing import Any, Dict, List, Optional
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL
from .tools import TOOLS

PLANNER_PROMPT = """You are a helpful assistant with tools.
Decide the NEXT best action. Output strict JSON only.

Tools:
{tools}

Rules:
- If answering from policy docs, use tool "rag_answer" with {{ "query": "..." }}.
- If user asks to request PTO/leave, use "create_leave_request".
- If user asks to approve a leave, use "approve_leave_request" with {{ "id": "<uuid>" }}.
- If user asks to reject a leave, use "reject_leave_request" with {{ "id": "<uuid>" }}.
- If user asks to list leave requests, use "list_leave_requests" with {{ "user": "<name>", "status": "<status>" }} if provided.
- If user asks to cancel a leave request, use "cancel_leave_request" with {{ "id": "<uuid>" }}.
- If asking about a specific date being a holiday, use "check_holiday".
- If user explicitly asks to call an API endpoint, use "http_get" or "http_post".
- If the answer is obvious and needs no tool, respond with {{ "action": "final", "answer": "..." }}.

Return JSON like:
{{ "action":"tool","name":"rag_answer","args":{{"query":"How many PTO days in Year 1?"}} }}
OR
{{ "action":"final","answer":"..." }}

User: {user_msg}
"""

SYNTH_PROMPT = """You ran one or more tools. Given the user message and the most recent tool output, write a final answer.
- Be concise (1-3 sentences).
- If citing policy facts, quote numbers/dates and add (source — section) if available.
Return strict JSON: {{ "final_answer": "..." }}.

User: {user_msg}
Most recent tool name: {tool_name}
Most recent tool output (JSON): {tool_output}
"""

def _llm():
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)

def _tools_description():
    lines = []
    for name, spec in TOOLS.items():
        lines.append(f"- {name}: {spec['description']}; schema={json.dumps(spec['schema'])}")
    return "\n".join(lines)

def plan_step(user_msg: str) -> Dict[str, Any]:
    llm = _llm()
    prompt = PLANNER_PROMPT.format(tools=_tools_description(), user_msg=user_msg)
    out = llm.invoke(prompt).content.strip()
    try:
        return json.loads(out)
    except Exception:
        return {"action":"final","answer":"Sorry, I couldn't parse a plan. Please rephrase."}

def synthesize(user_msg: str, tool_name: str, tool_output: Dict[str, Any]) -> str:
    llm = _llm()
    prompt = SYNTH_PROMPT.format(
        user_msg=user_msg,
        tool_name=tool_name,
        tool_output=json.dumps(tool_output, ensure_ascii=False)[:6000],
    )
    out = llm.invoke(prompt).content.strip()
    try:
        return json.loads(out).get("final_answer","")
    except Exception:
        return "Here is what I found: " + str(tool_output)[:500]

def run_agent(user_msg: str, max_steps: int = 3) -> Dict[str, Any]:
    """
    Returns:
      {
        "type": "final",
        "answer": "...",
        "steps": int,
        "trace": [
          {"step":1,"plan":{...}},
          {"step":1,"tool_call":{"name":"...","args":{...}}},
          {"step":1,"observation":{...}},
          ...
        ]
      }
    """
    trace: List[Dict[str, Any]] = []
    last_obs: Optional[Dict[str, Any]] = None
    last_tool_name: Optional[str] = None
    context_for_planner = user_msg

    for step in range(1, max_steps + 1):
        plan = plan_step(context_for_planner)
        trace.append({"step": step, "plan": plan})

        if plan.get("action") == "final":
            return {"type": "final", "answer": plan.get("answer",""), "steps": step, "trace": trace}

        if plan.get("action") == "tool":
            name = plan.get("name")
            args = plan.get("args", {})
            tool = TOOLS.get(name)
            if not tool:
                err = {"error": f"Unknown tool '{name}'."}
                trace.append({"step": step, "tool_call": {"name": name, "args": args}})
                trace.append({"step": step, "observation": err})
                return {"type": "final", "answer": err["error"], "steps": step, "trace": trace}

            trace.append({"step": step, "tool_call": {"name": name, "args": args}})
            try:
                obs = tool["fn"](args)
            except Exception as e:
                obs = {"error": str(e)}

            trace.append({"step": step, "observation": obs})
            last_obs = obs
            last_tool_name = name
            # feed observation back to planner context
            context_for_planner = f"{user_msg}\n(Previous result: {json.dumps(obs, ensure_ascii=False)[:2000]})"
            continue

        # planner returned something unexpected
        err = "Planner returned an unsupported action."
        trace.append({"step": step, "error": err})
        return {"type": "final", "answer": err, "steps": step, "trace": trace}

    # step limit reached → synthesize with last observation if any
    if last_obs is not None and last_tool_name is not None:
        final = synthesize(user_msg, last_tool_name, last_obs)
        trace.append({"synthesis": {"from_tool": last_tool_name}})
        return {"type": "final", "answer": final, "steps": max_steps, "trace": trace}

    return {"type":"final","answer":"I couldn't decide on a next action.","steps": max_steps, "trace": trace}

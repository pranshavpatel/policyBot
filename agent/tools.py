# agent/tools.py
from typing import Any, Dict, Callable, List, Optional
import functools
import requests
from tools.doc_search import doc_search
from tools.holiday_check import check_holiday
from tools.leave_request import (
    create_leave_request, list_leave_requests, get_leave_request,
    approve_leave_request, reject_leave_request, cancel_leave_request
)
from tools.qa_chain import build_qa_chain, ask
from auth.authz import assert_can_create, assert_can_moderate, assert_can_cancel, PermissionDenied
from auth.audit import write_audit

ToolFn = Callable[[Dict[str, Any]], Dict[str, Any]]

# Tools called from the chat agent (agent/react_agent.py) that touch leave
# requests receive the authenticated caller as args["_actor"] = {"username",
# "role"} rather than trusting a "user" field from the LLM's plan JSON — see
# ACTOR_SCOPED_TOOLS in agent/react_agent.py. This mirrors the same
# auth/authz.py rules the REST API enforces (api/app.py), so "ask the bot to
# approve my own leave" is rejected the same way a direct API call is.

def _require_actor(args: Dict[str, Any]) -> Dict[str, str]:
    actor = args.get("_actor")
    if not actor or not actor.get("username"):
        raise PermissionDenied("Sign in required for this action.")
    return actor


def _as_tool_error(fn: ToolFn) -> ToolFn:
    """Turns a PermissionDenied raised anywhere in an actor-scoped tool into
    the same {"error": "..."} shape every other tool failure already uses
    (e.g. "request not found" below), instead of letting it escape as a raw
    exception that only run_agent's catch-all would translate."""
    @functools.wraps(fn)
    def wrapper(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return fn(args)
        except PermissionDenied as e:
            return {"error": str(e)}
    return wrapper

# --- Safe HTTP tool (allowlist) ---
ALLOW_HTTP = ("http://localhost:8000", "https://httpbin.org")

def _clean_user(u):
    if not u:
        return None
    u = u.strip()
    # normalize typographic apostrophes
    u = u.replace("’", "'")
    # strip possessive like "Alice's" or "Alice 's"
    if u.endswith("'s"):
        u = u[:-2]
    if u.endswith(" 's"):
        u = u[:-3]
    return u.strip()

STATUS_ALIASES = {
    "pending": "submitted",
    "open": "submitted",
    "submit": "submitted",
    "submitted": "submitted",
    "approved": "approved",
    "rejected": "rejected",
    "canceled": "cancelled",
    "cancelled": "cancelled",
}

def http_get(args: Dict[str, Any]) -> Dict[str, Any]:
    url = args.get("url"); params = args.get("params") or {}
    if not url or not url.startswith(ALLOW_HTTP):
        return {"error": "URL not allowed"}
    r = requests.get(url, params=params, timeout=15)
    return {"status": r.status_code, "json": _safe_json(r)}

def http_post(args: Dict[str, Any]) -> Dict[str, Any]:
    url = args.get("url"); payload = args.get("json") or {}
    if not url or not url.startswith(ALLOW_HTTP):
        return {"error": "URL not allowed"}
    r = requests.post(url, json=payload, timeout=15)
    return {"status": r.status_code, "json": _safe_json(r)}

def _safe_json(r: requests.Response):
    try:
        return r.json()
    except Exception:
        return {"text": r.text[:2000]}

# --- Local tools wired to your project ---
def tool_doc_search(args: Dict[str, Any]) -> Dict[str, Any]:
    q = args.get("query", "")
    return doc_search(q, k=int(args.get("k", 5)))

@_as_tool_error
def tool_create_leave(args: Dict[str, Any]) -> Dict[str, Any]:
    actor = _require_actor(args)
    # Always create on behalf of the authenticated caller, never a "user"
    # field the LLM might have picked up from free text.
    assert_can_create(actor["username"], actor["username"])
    req = create_leave_request(
        user=actor["username"],
        start_date=args["start_date"],
        end_date=args["end_date"],
        reason=args["reason"],
    )
    write_audit(actor=actor["username"], action="create", target_type="leave_request", target_id=req["id"])
    return {"created": req}


@_as_tool_error
def tool_list_leave(args: Dict[str, Any]) -> Dict[str, Any]:
    actor = _require_actor(args)
    user = _clean_user(args.get("user"))
    # Employees may only ever list their own requests.
    if actor["role"] != "manager":
        user = actor["username"]
    status = args.get("status")
    if isinstance(status, str):
        status = STATUS_ALIASES.get(status.lower().strip(), status.lower().strip())
    rows = list_leave_requests(user=user, status=status)
    return {"requests": rows, "count": len(rows)}

@_as_tool_error
def tool_approve_leave(args: Dict[str, Any]) -> Dict[str, Any]:
    actor = _require_actor(args)
    row = get_leave_request(args["id"])
    if not row:
        return {"error": "request not found", "id": args["id"]}
    assert_can_moderate(actor["username"], actor["role"], row["user"])
    out = approve_leave_request(args["id"])
    write_audit(actor=actor["username"], action="approve", target_type="leave_request", target_id=args["id"])
    return out

@_as_tool_error
def tool_reject_leave(args: Dict[str, Any]) -> Dict[str, Any]:
    actor = _require_actor(args)
    row = get_leave_request(args["id"])
    if not row:
        return {"error": "request not found", "id": args["id"]}
    assert_can_moderate(actor["username"], actor["role"], row["user"])
    out = reject_leave_request(args["id"])
    write_audit(actor=actor["username"], action="reject", target_type="leave_request", target_id=args["id"])
    return out

@_as_tool_error
def tool_cancel_leave(args: Dict[str, Any]) -> Dict[str, Any]:
    actor = _require_actor(args)
    row = get_leave_request(args["id"])
    if not row:
        return {"error": "request not found", "id": args["id"]}
    assert_can_cancel(actor["username"], actor["role"], row["user"])
    out = cancel_leave_request(args["id"])
    write_audit(actor=actor["username"], action="cancel", target_type="leave_request", target_id=args["id"])
    return out

def tool_check_holiday(args: Dict[str, Any]) -> Dict[str, Any]:
    return {"result": check_holiday(args["date_str"])}

# --- Tool registry (name -> (description, schema, fn)) ---
TOOLS: Dict[str, Dict[str, Any]] = {
    "doc_search": {
        "description": "Search HR/IT policies and return snippets + citations.",
        "schema": {"type":"object","properties":{"query":{"type":"string"},"k":{"type":"integer"}},
                   "required":["query"]},
        "fn": tool_doc_search,
    },
    "create_leave_request": {
        "description": "Create a leave request for the current signed-in user (start_date, end_date, reason).",
        "schema": {"type":"object","properties":{
            "start_date":{"type":"string"},
            "end_date":{"type":"string"},"reason":{"type":"string"}},
            "required":["start_date","end_date","reason"]},
        "fn": tool_create_leave,
    },
    "check_holiday": {
        "description": "Check if a date (YYYY-MM-DD) is a holiday.",
        "schema": {"type":"object","properties":{"date_str":{"type":"string"}},
                   "required":["date_str"]},
        "fn": tool_check_holiday,
    },
    "http_get": {
        "description": "HTTP GET to allowed URLs (localhost:8000, httpbin.org).",
        "schema": {"type":"object","properties":{"url":{"type":"string"},"params":{"type":"object"}},
                   "required":["url"]},
        "fn": http_get,
    },
    "http_post": {
        "description": "HTTP POST to allowed URLs (localhost:8000).",
        "schema": {"type":"object","properties":{"url":{"type":"string"},"json":{"type":"object"}},
                   "required":["url"]},
        "fn": http_post,
    },
}

_qa = build_qa_chain(k=5)
def tool_rag_answer(args: Dict[str, Any]) -> Dict[str, Any]:
    q = args.get("query", "")
    ans, srcs = ask(_qa, q)
    cites = [{"source": d.metadata.get("source"),
              "section": d.metadata.get("h2") or d.metadata.get("h1") or d.metadata.get("h3","")} for d in srcs]
    return {"answer": ans, "citations": cites}

# register it
TOOLS["rag_answer"] = {
    "description": "Answer HR policy questions using RAG with citations.",
    "schema": {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]},
    "fn": tool_rag_answer,
}

TOOLS["list_leave_requests"] = {
    "description": "List leave requests; optional filters: user, status(submitted|approved|rejected|cancelled).",
    "schema": {"type":"object","properties":{"user":{"type":"string"},"status":{"type":"string"}}, "required":[]},
    "fn": tool_list_leave,
}

TOOLS["cancel_leave_request"] = {
    "description": "Cancel a leave request by id.",
    "schema": {"type":"object","properties":{"id":{"type":"string"}},"required":["id"]},
    "fn": tool_cancel_leave,
}
TOOLS["approve_leave_request"] = {
    "description": "Approve a leave request by id.",
    "schema": {"type":"object","properties":{"id":{"type":"string"}},"required":["id"]},
    "fn": tool_approve_leave,
}
TOOLS["reject_leave_request"] = {
    "description": "Reject a leave request by id.",
    "schema": {"type":"object","properties":{"id":{"type":"string"}},"required":["id"]},
    "fn": tool_reject_leave,
}
"""Unit tests for the actor-scoped chat-agent tools (agent/tools.py).

These call the tool functions directly with an injected _actor, the same
way agent/react_agent.py's run_agent() does for ACTOR_SCOPED_TOOLS — this
is deliberately decoupled from testing the LLM's JSON planning output
(which would require mocking ChatGroq's response format and mostly test
prompt-following, not our authorization wiring)."""
from agent.tools import (
    tool_create_leave, tool_list_leave, tool_approve_leave,
    tool_reject_leave, tool_cancel_leave,
)

ALICE = {"username": "alice", "role": "employee"}
BOB = {"username": "bob", "role": "employee"}
MANAGER = {"username": "manager1", "role": "manager"}


def test_create_leave_forces_owner_to_actor():
    out = tool_create_leave({
        "_actor": ALICE, "start_date": "2026-11-01", "end_date": "2026-11-02", "reason": "trip",
    })
    assert out["created"]["user"] == "alice"


def test_create_leave_requires_actor():
    out = tool_create_leave({"start_date": "2026-11-01", "end_date": "2026-11-02", "reason": "trip"})
    assert "error" in out


def test_list_leave_scopes_employee_to_self():
    tool_create_leave({"_actor": BOB, "start_date": "2026-12-01", "end_date": "2026-12-02", "reason": "trip"})
    out = tool_list_leave({"_actor": BOB, "user": "manager1"})  # tries to peek at someone else
    assert all(r["user"] == "bob" for r in out["requests"])


def test_manager_can_approve_others_not_self():
    created = tool_create_leave({"_actor": ALICE, "start_date": "2027-01-01", "end_date": "2027-01-02", "reason": "pto"})
    req_id = created["created"]["id"]

    approved = tool_approve_leave({"_actor": MANAGER, "id": req_id})
    assert approved["ok"] is True

    own = tool_create_leave({"_actor": MANAGER, "start_date": "2027-02-01", "end_date": "2027-02-02", "reason": "pto"})
    self_approve = tool_approve_leave({"_actor": MANAGER, "id": own["created"]["id"]})
    assert "error" in self_approve


def test_employee_cannot_reject_anyones_request():
    created = tool_create_leave({"_actor": BOB, "start_date": "2027-03-01", "end_date": "2027-03-02", "reason": "pto"})
    out = tool_reject_leave({"_actor": ALICE, "id": created["created"]["id"]})
    assert "error" in out


def test_owner_can_cancel_own_request():
    created = tool_create_leave({"_actor": ALICE, "start_date": "2027-04-01", "end_date": "2027-04-02", "reason": "pto"})
    out = tool_cancel_leave({"_actor": ALICE, "id": created["created"]["id"]})
    assert out["ok"] is True
    assert out["request"]["status"] == "cancelled"

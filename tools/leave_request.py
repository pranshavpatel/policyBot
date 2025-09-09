# tools/leave_request.py
"""
Leave Request tool (simulated).
- create_leave_request(user, start_date, end_date, reason) -> dict
- list_leave_requests(user=None) -> list[dict]
- get_leave_request(req_id) -> dict | None

Storage: data/requests_db.json  (an array of request objects)
Date format: ISO 'YYYY-MM-DD'
"""

# tools/leave_request.py  (DB-backed version)
from __future__ import annotations
from datetime import datetime
import uuid
from typing import Optional, List, Dict, Any
from sqlalchemy import select
from db.session import SessionLocal
from db.models import LeaveRequest

VALID_STATUSES = {"submitted", "approved", "rejected", "cancelled"}

def _validate(user: str, start_date: str, end_date: str, reason: str):
    if not user: raise ValueError("user required")
    if not reason: raise ValueError("reason required")
    sd = datetime.fromisoformat(start_date).date()
    ed = datetime.fromisoformat(end_date).date()
    if ed < sd: raise ValueError("end_date must be on or after start_date")
    return sd, ed

def create_leave_request(user: str, start_date: str, end_date: str, reason: str) -> Dict[str, Any]:
    sd, ed = _validate(user, start_date, end_date, reason)
    with SessionLocal() as s:
        req = LeaveRequest(
            id=str(uuid.uuid4()),
            user=user,
            start_date=sd,
            end_date=ed,
            reason=reason,
            status="submitted",
            created_at=datetime.utcnow(),
        )
        s.add(req)
        s.commit()
        return {
            "id": req.id, "user": req.user, "start_date": start_date, "end_date": end_date,
            "reason": req.reason, "status": req.status, "created_at": req.created_at.isoformat()+"Z"
        }

def list_leave_requests(user: Optional[str]=None, status: Optional[str]=None) -> List[Dict[str, Any]]:
    with SessionLocal() as s:
        stmt = select(LeaveRequest)
        if user:  stmt = stmt.filter(LeaveRequest.user == user)
        if status:
            if status not in VALID_STATUSES: raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
            stmt = stmt.filter(LeaveRequest.status == status)
        rows = s.execute(stmt.order_by(LeaveRequest.created_at.desc())).scalars().all()
        out=[]
        for r in rows:
            out.append({
                "id": r.id, "user": r.user,
                "start_date": r.start_date.isoformat(),
                "end_date": r.end_date.isoformat(),
                "reason": r.reason, "status": r.status,
                "created_at": r.created_at.isoformat()+"Z",
                "updated_at": r.updated_at.isoformat()+"Z" if r.updated_at else None
            })
        return out

def get_leave_request(req_id: str) -> Optional[Dict[str, Any]]:
    with SessionLocal() as s:
        r = s.get(LeaveRequest, req_id)
        if not r: return None
        return {
            "id": r.id, "user": r.user,
            "start_date": r.start_date.isoformat(),
            "end_date": r.end_date.isoformat(),
            "reason": r.reason, "status": r.status,
            "created_at": r.created_at.isoformat()+"Z",
            "updated_at": r.updated_at.isoformat()+"Z" if r.updated_at else None
        }

def _set_status(req_id: str, new_status: str) -> Dict[str, Any]:
    if new_status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
    with SessionLocal() as s:
        r = s.get(LeaveRequest, req_id)
        if not r:
            return {"ok": False, "error": "request not found", "id": req_id}
        r.status = new_status
        r.updated_at = datetime.utcnow()
        s.commit()
        return {"ok": True, "request": get_leave_request(req_id), "previous_status": None}

def approve_leave_request(req_id: str) -> Dict[str, Any]:
    return _set_status(req_id, "approved")

def reject_leave_request(req_id: str) -> Dict[str, Any]:
    return _set_status(req_id, "rejected")

def cancel_leave_request(req_id: str) -> Dict[str, Any]:
    return _set_status(req_id, "cancelled")

# ---------- (optional) JSON schema for an LLM planner/validator later

CREATE_LEAVE_REQUEST_SCHEMA: Dict[str, Any] = {
    "name": "create_leave_request",
    "description": "Create a PTO/leave request in the HR system (simulated).",
    "schema": {
        "type": "object",
        "properties": {
            "user": {"type": "string", "description": "username or email"},
            "start_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            "reason": {"type": "string", "description": "short reason for leave"},
        },
        "required": ["user", "start_date", "end_date", "reason"],
        "additionalProperties": False,
    },
}

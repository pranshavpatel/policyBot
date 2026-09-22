"""Append-only audit log for every leave-request state change."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from db.session import SessionLocal
from db.models import AuditLog


def write_audit(actor: str, action: str, target_type: str, target_id: str, detail: Optional[str] = None) -> None:
    with SessionLocal() as s:
        s.add(AuditLog(
            id=str(uuid.uuid4()),
            actor=actor,
            action=action,
            target_type=target_type,
            target_id=target_id,
            detail=detail,
            created_at=datetime.utcnow(),
        ))
        s.commit()


def list_audit(limit: int = 200) -> List[Dict[str, Any]]:
    with SessionLocal() as s:
        rows = (
            s.query(AuditLog)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "actor": r.actor,
                "action": r.action,
                "target_type": r.target_type,
                "target_id": r.target_id,
                "detail": r.detail,
                "created_at": r.created_at.isoformat() + "Z",
            }
            for r in rows
        ]

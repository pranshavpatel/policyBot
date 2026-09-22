"""
Seeded demo user directory.

This is a portfolio project, not a real HR system, so users are an in-memory
seed rather than a full identity provider. Swapping this for a real user
table / SSO integration would be a drop-in change: everything downstream
only depends on the (username, role) shape returned by get_user().

Roles:
  - "employee": can create/view/cancel their own leave requests, ask
    policy questions.
  - "manager":  can additionally approve/reject/list ANY employee's leave
    requests (but not their own — see auth/authz.py), and read the audit
    log.

Demo credentials (also documented in README):
  alice    / alice123     (employee)
  bob      / bob123       (employee)
  manager1 / manager123   (manager)
"""
from __future__ import annotations
from typing import Dict, Optional, TypedDict

from auth.security import hash_password


class User(TypedDict):
    username: str
    role: str
    hashed_password: str


_SEED = [
    ("alice", "employee", "alice123"),
    ("bob", "employee", "bob123"),
    ("manager1", "manager", "manager123"),
]

USERS: Dict[str, User] = {
    username: {"username": username, "role": role, "hashed_password": hash_password(pw)}
    for username, role, pw in _SEED
}


def get_user(username: str) -> Optional[User]:
    return USERS.get(username)

"""Authorization rules, kept as small pure functions so both the REST API
and the chat agent enforce the exact same policy."""
from __future__ import annotations


class PermissionDenied(Exception):
    pass


def assert_can_view(actor_username: str, actor_role: str, target_user: str) -> None:
    if actor_role == "manager":
        return
    if actor_username != target_user:
        raise PermissionDenied(f"{actor_username} may only view their own leave requests")


def assert_can_create(actor_username: str, requested_user: str) -> None:
    if actor_username != requested_user:
        raise PermissionDenied("You can only create a leave request for yourself")


def assert_can_moderate(actor_username: str, actor_role: str, request_owner: str) -> None:
    """Approve/reject. Managers only, and never their own request."""
    if actor_role != "manager":
        raise PermissionDenied("Only managers can approve or reject leave requests")
    if actor_username == request_owner:
        raise PermissionDenied("Managers cannot approve or reject their own leave request")


def assert_can_cancel(actor_username: str, actor_role: str, request_owner: str) -> None:
    if actor_role == "manager":
        return
    if actor_username != request_owner:
        raise PermissionDenied("You can only cancel your own leave request")

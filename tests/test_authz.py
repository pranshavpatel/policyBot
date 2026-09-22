import pytest
from auth.authz import (
    assert_can_view, assert_can_create, assert_can_moderate, assert_can_cancel,
    PermissionDenied,
)


def test_employee_can_view_own():
    assert_can_view("alice", "employee", "alice")


def test_employee_cannot_view_others():
    with pytest.raises(PermissionDenied):
        assert_can_view("alice", "employee", "bob")


def test_manager_can_view_anyone():
    assert_can_view("manager1", "manager", "alice")


def test_employee_can_only_create_for_self():
    assert_can_create("alice", "alice")
    with pytest.raises(PermissionDenied):
        assert_can_create("alice", "bob")


def test_manager_can_moderate_others_request():
    assert_can_moderate("manager1", "manager", "alice")


def test_manager_cannot_moderate_own_request():
    with pytest.raises(PermissionDenied):
        assert_can_moderate("manager1", "manager", "manager1")


def test_employee_cannot_moderate_at_all():
    with pytest.raises(PermissionDenied):
        assert_can_moderate("alice", "employee", "bob")


def test_cancel_self_or_manager_only():
    assert_can_cancel("alice", "employee", "alice")
    assert_can_cancel("manager1", "manager", "alice")
    with pytest.raises(PermissionDenied):
        assert_can_cancel("alice", "employee", "bob")

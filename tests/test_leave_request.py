import pytest
from tools.leave_request import (
    create_leave_request, list_leave_requests, get_leave_request,
    approve_leave_request, reject_leave_request, cancel_leave_request,
)


def test_create_and_get_round_trip():
    req = create_leave_request("carol", "2026-01-05", "2026-01-07", "vacation")
    assert req["status"] == "submitted"
    fetched = get_leave_request(req["id"])
    assert fetched["user"] == "carol"
    assert fetched["start_date"] == "2026-01-05"


def test_end_before_start_rejected():
    with pytest.raises(ValueError):
        create_leave_request("carol", "2026-01-10", "2026-01-05", "oops")


def test_missing_reason_rejected():
    with pytest.raises(ValueError):
        create_leave_request("carol", "2026-01-05", "2026-01-07", "")


def test_list_filters_by_user_and_status():
    create_leave_request("dave", "2026-02-01", "2026-02-02", "sick")
    req2 = create_leave_request("dave", "2026-03-01", "2026-03-02", "vacation")
    approve_leave_request(req2["id"])

    all_dave = list_leave_requests(user="dave")
    assert len(all_dave) >= 2

    approved_dave = list_leave_requests(user="dave", status="approved")
    assert all(r["status"] == "approved" for r in approved_dave)
    assert any(r["id"] == req2["id"] for r in approved_dave)


def test_approve_reject_cancel_transitions():
    req = create_leave_request("erin", "2026-04-01", "2026-04-02", "conference")
    out = approve_leave_request(req["id"])
    assert out["ok"] is True
    assert out["request"]["status"] == "approved"

    req2 = create_leave_request("erin", "2026-05-01", "2026-05-02", "conference")
    out2 = reject_leave_request(req2["id"])
    assert out2["request"]["status"] == "rejected"

    req3 = create_leave_request("erin", "2026-06-01", "2026-06-02", "conference")
    out3 = cancel_leave_request(req3["id"])
    assert out3["request"]["status"] == "cancelled"


def test_set_status_on_missing_id_returns_error_not_exception():
    out = approve_leave_request("does-not-exist")
    assert out["ok"] is False
    assert "not found" in out["error"]


def test_invalid_status_filter_raises():
    with pytest.raises(ValueError):
        list_leave_requests(status="not-a-real-status")

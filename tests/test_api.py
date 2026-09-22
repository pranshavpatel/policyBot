"""API-level tests via FastAPI's TestClient. Deliberately avoid /chat and
/agent here — those call the real LLM (see tests/test_agent.py, which mocks
it instead of skipping the code path entirely)."""


def test_login_success_and_failure(client):
    ok = client.post("/auth/login", data={"username": "alice", "password": "alice123"})
    assert ok.status_code == 200
    body = ok.json()
    assert body["role"] == "employee"
    assert body["access_token"]

    bad = client.post("/auth/login", data={"username": "alice", "password": "wrong"})
    assert bad.status_code == 401


def test_leave_requests_require_auth(client):
    r = client.post("/leave-requests", json={"start_date": "2026-01-01", "end_date": "2026-01-02", "reason": "x"})
    assert r.status_code == 401


def test_create_list_own_request(client, auth_headers):
    headers = client.post("/auth/login", data={"username": "bob", "password": "bob123"})
    headers = {"Authorization": f"Bearer {headers.json()['access_token']}"}

    created = client.post("/leave-requests", json={
        "start_date": "2026-07-01", "end_date": "2026-07-03", "reason": "trip",
    }, headers=headers)
    assert created.status_code == 200
    assert created.json()["user"] == "bob"

    listed = client.get("/leave-requests", headers=headers)
    assert listed.status_code == 200
    assert all(r["user"] == "bob" for r in listed.json())


def test_employee_cannot_see_others_via_query_param(client, auth_headers):
    alice_headers = auth_headers("alice", "employee")
    listed = client.get("/leave-requests?user=manager1", headers=alice_headers)
    assert listed.status_code == 200
    # forced back to alice's own requests regardless of ?user=
    assert all(r["user"] == "alice" for r in listed.json())


def test_manager_can_approve_others_but_not_own(client, auth_headers):
    alice_headers = auth_headers("alice", "employee")
    manager_headers = auth_headers("manager1", "manager")

    created = client.post("/leave-requests", json={
        "start_date": "2026-08-01", "end_date": "2026-08-02", "reason": "pto",
    }, headers=alice_headers).json()

    approved = client.post(f"/leave-requests/{created['id']}/approve", headers=manager_headers)
    assert approved.status_code == 200
    assert approved.json()["request"]["status"] == "approved"

    own_request = client.post("/leave-requests", json={
        "start_date": "2026-09-01", "end_date": "2026-09-02", "reason": "pto",
    }, headers=manager_headers).json()
    self_approve = client.post(f"/leave-requests/{own_request['id']}/approve", headers=manager_headers)
    assert self_approve.status_code == 403


def test_employee_cannot_approve_anyones_request(client, auth_headers):
    alice_headers = auth_headers("alice", "employee")
    bob_headers = auth_headers("bob", "employee")

    created = client.post("/leave-requests", json={
        "start_date": "2026-10-01", "end_date": "2026-10-02", "reason": "pto",
    }, headers=bob_headers).json()

    r = client.post(f"/leave-requests/{created['id']}/approve", headers=alice_headers)
    assert r.status_code == 403


def test_audit_log_requires_manager(client, auth_headers):
    alice_headers = auth_headers("alice", "employee")
    manager_headers = auth_headers("manager1", "manager")

    forbidden = client.get("/audit-log", headers=alice_headers)
    assert forbidden.status_code == 403

    allowed = client.get("/audit-log", headers=manager_headers)
    assert allowed.status_code == 200
    assert isinstance(allowed.json(), list)


def test_holidays_next_route_not_shadowed(client):
    # Regression test: /holidays/next used to be unreachable because
    # /holidays/{date_str} was registered first and matched "next" as a
    # (invalid) date string.
    r = client.get("/holidays/next?n=2")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert all("name" in h for h in body)


def test_holidays_date_lookup_still_works(client):
    r = client.get("/holidays/2025-07-04")
    assert r.status_code == 200
    assert r.json()["is_holiday"] is True


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}

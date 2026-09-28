"""API-level tests via FastAPI's TestClient. Deliberately avoid making real
/chat or /agent calls (those hit the real LLM) — except for
test_agent_rate_limit_surfaces_as_429_not_500 below, which monkeypatches
run_agent to test the error-handling path around it without needing a real
LLM call."""


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


def test_agent_rate_limit_surfaces_as_429_not_500(client, auth_headers, monkeypatch):
    # Regression test: a live 70-question eval run hit Groq's rate limit
    # partway through, and without api/app.py's exception handler for
    # groq.RateLimitError, every subsequent request failed with an opaque
    # 500 (FastAPI's default handler hides the real cause) — indistinguishable
    # from an actual bug to a caller or to scripts/eval_via_api.py's retry
    # logic, which specifically needs a 429 to know to back off and retry.
    import httpx
    from groq import RateLimitError
    import api.app as app_module

    fake_response = httpx.Response(429, headers={"retry-after": "3"}, request=httpx.Request("POST", "http://x"))

    def _raise_rate_limit(*args, **kwargs):
        raise RateLimitError("rate limited", response=fake_response, body=None)

    monkeypatch.setattr(app_module, "run_agent", _raise_rate_limit)

    headers = auth_headers("alice", "employee")
    r = client.post("/agent", json={"message": "How many PTO days?", "trace": False}, headers=headers)
    assert r.status_code == 429
    assert r.headers.get("retry-after") == "3"

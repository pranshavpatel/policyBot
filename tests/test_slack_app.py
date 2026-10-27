"""Unit tests for slack/slack_app.py's non-Slack-SDK logic: logging in to
the PolicyBot API, retrying once on a 401, and formatting the response for
Slack. Mocked HTTP (no real PolicyBot API or Slack network calls) — the
module itself is importable in CI because token_verification_enabled=False
(see that module) skips the live Slack auth.test call App() would
otherwise make on import.

Verified live against a real Slack workspace and a locally running
PolicyBot API (both the rag_answer read path and the create_leave_request
write path) — see README's Slack section; this file covers what a live
run doesn't: retry-on-401 and the error path, which aren't easy to
trigger live on demand."""
import slack.slack_app as slack_app


class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def test_get_token_logs_in_and_caches(monkeypatch):
    calls = []

    def fake_post(url, data=None, timeout=None):
        calls.append(url)
        assert url.endswith("/auth/login")
        return _FakeResponse(200, {"access_token": "tok-1"})

    monkeypatch.setattr(slack_app.requests, "post", fake_post)
    slack_app._token_cache.update(value=None, expires_at=0)

    tok1 = slack_app._get_token()
    tok2 = slack_app._get_token()  # should hit the cache, not log in again

    assert tok1 == tok2 == "tok-1"
    assert len(calls) == 1


def test_call_agent_retries_once_on_401(monkeypatch):
    login_calls = {"n": 0}
    agent_calls = {"n": 0}

    def fake_post(url, headers=None, data=None, timeout=None):
        if url.endswith("/auth/login"):
            login_calls["n"] += 1
            return _FakeResponse(200, {"access_token": f"tok-{login_calls['n']}"})
        agent_calls["n"] += 1
        if agent_calls["n"] == 1:
            return _FakeResponse(401, {})  # stale cached token
        return _FakeResponse(200, {"type": "final", "answer": "15 days", "steps": 1})

    monkeypatch.setattr(slack_app.requests, "post", fake_post)
    slack_app._token_cache.update(value="stale-token", expires_at=slack_app.time.time() + 3600)

    result = slack_app.call_agent("How many PTO days?")

    assert result == {"type": "final", "answer": "15 days", "steps": 1}
    assert agent_calls["n"] == 2  # one 401, one retry that succeeded
    assert login_calls["n"] == 1  # only re-logged-in once, on the retry


def test_call_agent_returns_error_payload_on_exhausted_retries(monkeypatch):
    def fake_post(url, data=None, headers=None, timeout=None):
        if url.endswith("/auth/login"):
            return _FakeResponse(200, {"access_token": "tok"})
        return _FakeResponse(401, {})  # keeps failing even after re-login

    monkeypatch.setattr(slack_app.requests, "post", fake_post)
    slack_app._token_cache.update(value=None, expires_at=0)

    result = slack_app.call_agent("hello")

    assert result["type"] == "error"


def test_format_result_renders_answer_and_trace():
    payload = {"type": "final", "answer": "15 days", "trace": [{"step": 1}]}
    blocks = slack_app.format_result(payload)

    assert any("15 days" in b["text"]["text"] for b in blocks)
    assert any("Trace" in b["text"]["text"] for b in blocks)


def test_format_result_without_trace_has_no_trace_block():
    payload = {"type": "final", "answer": "hi"}
    blocks = slack_app.format_result(payload)

    assert len(blocks) == 1
    assert not any("Trace" in b["text"]["text"] for b in blocks)

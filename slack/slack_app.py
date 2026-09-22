import os, json, time, threading, requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
SHOW_TRACE_DEFAULT = os.getenv("SHOW_TRACE_DEFAULT", "false").lower() == "true"

# The API now requires a JWT (see auth/), not a static bearer token. Every
# Slack user currently acts through one shared "service" identity — this
# bot doesn't map individual Slack users to individual PolicyBot accounts,
# so all leave-request actions from Slack are attributed to this account in
# the audit log. Mapping real per-user identity (e.g. Slack SSO email ->
# PolicyBot username) would be the natural next step if this needs
# per-employee attribution from Slack specifically.
SERVICE_USERNAME = os.getenv("POLICYBOT_SERVICE_USERNAME", "manager1")
SERVICE_PASSWORD = os.getenv("POLICYBOT_SERVICE_PASSWORD", "manager123")

app = App(token=os.environ["SLACK_BOT_TOKEN"])

_token_lock = threading.Lock()
_token_cache = {"value": None, "expires_at": 0}


def _get_token(force_refresh: bool = False) -> str:
    with _token_lock:
        if not force_refresh and _token_cache["value"] and time.time() < _token_cache["expires_at"]:
            return _token_cache["value"]
        resp = requests.post(
            f"{API_BASE}/auth/login",
            data={"username": SERVICE_USERNAME, "password": SERVICE_PASSWORD},
            timeout=15,
        )
        resp.raise_for_status()
        token = resp.json()["access_token"]
        # Re-login well before the server-side expiry (see JWT_EXPIRE_MINUTES).
        _token_cache.update(value=token, expires_at=time.time() + 45 * 60)
        return token


def call_agent(message: str, trace: bool = False):
    for attempt in (False, True):  # retry once with a forced re-login on 401
        try:
            resp = requests.post(
                f"{API_BASE}/agent",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {_get_token(force_refresh=attempt)}"},
                data=json.dumps({"message": message, "trace": trace}),
                timeout=60,
            )
            if resp.status_code == 401 and not attempt:
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"type": "error", "answer": f"Agent error: {e}"}
    return {"type": "error", "answer": "Agent error: could not authenticate with PolicyBot API"}


def format_result(payload: dict):
    # Pretty output for Slack (markdown + code block for JSON bits)
    t = payload.get("type")
    ans = payload.get("answer", "")
    blocks = [
        {"type":"section","text":{"type":"mrkdwn","text":f"*PolicyBot* → `{t}`\n{ans}"}}
    ]
    # If tool observation present, render a compact JSON block
    if "observation" in payload:
        obs = json.dumps(payload["observation"], indent=2)
        blocks.append({"type":"section","text":{"type":"mrkdwn","text":f"```json\n{obs}\n```"}})
    # Trace (optional)
    if payload.get("trace"):
        trace = json.dumps(payload["trace"], indent=2)
        blocks.append({"type":"section","text":{"type":"mrkdwn","text":f"*Trace:*```json\n{trace}\n```"}})
    return blocks

@app.event("app_mention")
def on_mention(body, say):
    text = body.get("event", {}).get("text", "")
    trace = "trace" in text.lower() or SHOW_TRACE_DEFAULT
    result = call_agent(text, trace=trace)
    say(blocks=format_result(result))

@app.event("message")
def on_dm(message, say, context):
    # Respond to DMs (ignore channels unless mentioned)
    if message.get("channel_type") == "im" and "bot_id" not in message:
        text = message.get("text", "")
        trace = "trace" in text.lower() or SHOW_TRACE_DEFAULT
        result = call_agent(text, trace=trace)
        say(blocks=format_result(result))

@app.command("/policybot")
def on_slash(ack, respond, command):
    ack()
    text = command.get("text", "").strip() or "help"
    trace = "trace" in text.lower() or SHOW_TRACE_DEFAULT
    result = call_agent(text, trace=trace)
    respond(blocks=format_result(result))

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()

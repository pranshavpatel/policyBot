import os, json, requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "dev-token-123")
SHOW_TRACE_DEFAULT = os.getenv("SHOW_TRACE_DEFAULT", "false").lower() == "true"

app = App(token=os.environ["SLACK_BOT_TOKEN"])

def call_agent(message: str, trace: bool = False):
    try:
        resp = requests.post(
            f"{API_BASE}/agent",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_AUTH_TOKEN}"},
            data=json.dumps({"message": message, "trace": trace}),
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"type": "error", "answer": f"Agent error: {e}"}

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
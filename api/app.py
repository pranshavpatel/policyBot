# api/app.py (excerpt)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from tools.leave_request import create_leave_request, list_leave_requests, approve_leave_request, reject_leave_request, get_leave_request
from tools.holiday_check import check_holiday, list_holidays, next_holidays
from tools.qa_chain import build_qa_chain, ask
from agent.react_agent import run_agent
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PolicyBot API")

origins = [
    "http://localhost:5173",
    "https://pranshavpatel.vercel.app" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
qa = build_qa_chain(k=5)

class LeaveRequestIn(BaseModel):
    user: str
    start_date: str  # YYYY-MM-DD
    end_date: str
    reason: str

class ChatIn(BaseModel):
    user: str
    message: str

class AgentIn(BaseModel):
    user: str = "pranshav"
    message: str
    trace: bool = False

@app.post("/leave-requests")
def api_create_leave(req: LeaveRequestIn):
    return create_leave_request(req.user, req.start_date, req.end_date, req.reason)

@app.get("/leave-requests")
def api_list_leave(user: Optional[str] = None):
    return list_leave_requests(user)

@app.get("/leave-requests/{req_id}")
def api_get_leave(req_id: str):
    return get_leave_request(req_id)

@app.get("/holidays/{date_str}")
def api_check_holiday(date_str: str):
    return check_holiday(date_str)

@app.get("/holidays")
def api_list_all(year: Optional[int] = None):
    return list_holidays(year)

@app.get("/holidays/next")
def api_next(n: int = 5, start_date: Optional[str] = None):
    return next_holidays(n=n, start_date=start_date)

@app.post("/chat")
def api_chat(q: ChatIn):
    ans, srcs = ask(qa, q.message)
    cites = [(d.metadata.get("source"),
              d.metadata.get("h2") or d.metadata.get("h1") or d.metadata.get("h3",""))
             for d in srcs]
    return {"answer": ans, "citations": cites}

@app.post("/agent")
def api_agent(body: AgentIn):
    res = run_agent(body.message)
    return res if body.trace else {k: v for k, v in res.items() if k not in {"trace"}}

@app.get("/leave-requests")  # list with optional filters
def api_list_leave(user: Optional[str] = None, status: Optional[str] = None):
    return list_leave_requests(user=user, status=status)

@app.post("/leave-requests/{req_id}/approve")
def api_approve_leave(req_id: str):
    return approve_leave_request(req_id)

@app.post("/leave-requests/{req_id}/reject")
def api_reject_leave(req_id: str):
    return reject_leave_request(req_id)

@app.get("/health")
def health():
    return {"status": "ok"}
# api/app.py
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional

from tools.leave_request import (
    create_leave_request, list_leave_requests, approve_leave_request,
    reject_leave_request, cancel_leave_request, get_leave_request,
)
from tools.holiday_check import check_holiday, list_holidays, next_holidays
from tools.qa_chain import build_qa_chain, ask
from agent.react_agent import run_agent

from auth.deps import get_current_user, require_manager, CurrentUser
from auth.users import get_user
from auth.security import verify_password, create_access_token
from auth.authz import assert_can_view, assert_can_create, assert_can_moderate, assert_can_cancel, PermissionDenied
from auth.audit import write_audit, list_audit

from db.session import Base, engine
from db import models  # noqa: F401 (ensures models are registered before create_all)

from observability import RequestLoggingMiddleware, get_logger

log = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Idempotent; lets the API run standalone without a manual db_init step.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="PolicyBot API", lifespan=lifespan)

origins = [
    "http://localhost:5173",
    "https://pranshavpatel.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

qa = build_qa_chain(k=5)


def _forbidden(e: PermissionDenied):
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


class LeaveRequestIn(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str
    reason: str


class ChatIn(BaseModel):
    message: str


class AgentIn(BaseModel):
    message: str
    trace: bool = False


# --- Auth ---

@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form.username)
    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    token = create_access_token(subject=user["username"], role=user["role"])
    write_audit(actor=user["username"], action="login", target_type="auth", target_id=user["username"])
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}


# --- Leave requests (authenticated) ---

@app.post("/leave-requests")
def api_create_leave(req: LeaveRequestIn, me: CurrentUser = Depends(get_current_user)):
    try:
        assert_can_create(me["username"], me["username"])
    except PermissionDenied as e:
        _forbidden(e)
    out = create_leave_request(me["username"], req.start_date, req.end_date, req.reason)
    write_audit(actor=me["username"], action="create", target_type="leave_request", target_id=out["id"])
    return out


@app.get("/leave-requests")
def api_list_leave(user: Optional[str] = None, status: Optional[str] = None,
                    me: CurrentUser = Depends(get_current_user)):
    # Employees can only ever see their own requests, regardless of the
    # `user` query param; managers can see anyone's.
    if me["role"] != "manager":
        user = me["username"]
    return list_leave_requests(user=user, status=status)


@app.get("/leave-requests/{req_id}")
def api_get_leave(req_id: str, me: CurrentUser = Depends(get_current_user)):
    row = get_leave_request(req_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    try:
        assert_can_view(me["username"], me["role"], row["user"])
    except PermissionDenied as e:
        _forbidden(e)
    return row


@app.post("/leave-requests/{req_id}/approve")
def api_approve_leave(req_id: str, me: CurrentUser = Depends(get_current_user)):
    row = get_leave_request(req_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    try:
        assert_can_moderate(me["username"], me["role"], row["user"])
    except PermissionDenied as e:
        _forbidden(e)
    out = approve_leave_request(req_id)
    write_audit(actor=me["username"], action="approve", target_type="leave_request", target_id=req_id)
    return out


@app.post("/leave-requests/{req_id}/reject")
def api_reject_leave(req_id: str, me: CurrentUser = Depends(get_current_user)):
    row = get_leave_request(req_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    try:
        assert_can_moderate(me["username"], me["role"], row["user"])
    except PermissionDenied as e:
        _forbidden(e)
    out = reject_leave_request(req_id)
    write_audit(actor=me["username"], action="reject", target_type="leave_request", target_id=req_id)
    return out


@app.post("/leave-requests/{req_id}/cancel")
def api_cancel_leave(req_id: str, me: CurrentUser = Depends(get_current_user)):
    row = get_leave_request(req_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    try:
        assert_can_cancel(me["username"], me["role"], row["user"])
    except PermissionDenied as e:
        _forbidden(e)
    out = cancel_leave_request(req_id)
    write_audit(actor=me["username"], action="cancel", target_type="leave_request", target_id=req_id)
    return out


# --- Audit log (managers only) ---

@app.get("/audit-log")
def api_audit_log(limit: int = 200, me: CurrentUser = Depends(require_manager)):
    return list_audit(limit=limit)


# --- Holidays (read-only, no policy-relevant secrecy — left open) ---

@app.get("/holidays/next")
def api_next(n: int = 5, start_date: Optional[str] = None):
    return next_holidays(n=n, start_date=start_date)


@app.get("/holidays/{date_str}")
def api_check_holiday(date_str: str):
    return check_holiday(date_str)


@app.get("/holidays")
def api_list_all(year: Optional[int] = None):
    return list_holidays(year)


# --- Chat / agent (authenticated so tool calls run as a real identity) ---

@app.post("/chat")
def api_chat(q: ChatIn, me: CurrentUser = Depends(get_current_user)):
    ans, srcs = ask(qa, q.message)
    cites = [(d.metadata.get("source"),
              d.metadata.get("h2") or d.metadata.get("h1") or d.metadata.get("h3", ""))
             for d in srcs]
    return {"answer": ans, "citations": cites}


@app.post("/agent")
def api_agent(body: AgentIn, me: CurrentUser = Depends(get_current_user)):
    res = run_agent(body.message, actor=me)
    return res if body.trace else {k: v for k, v in res.items() if k not in {"trace"}}


@app.get("/health")
def health():
    return {"status": "ok"}

"""FastAPI dependencies for authentication."""
from __future__ import annotations
from typing import TypedDict

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from auth.security import decode_access_token, JWTError
from auth.users import get_user

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


class CurrentUser(TypedDict):
    username: str
    role: str


def get_current_user(token: str = Depends(oauth2_scheme)) -> CurrentUser:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
    except JWTError:
        raise unauthorized
    username = payload.get("sub")
    role = payload.get("role")
    if not username or not role or not get_user(username):
        raise unauthorized
    return {"username": username, "role": role}


def require_manager(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user["role"] != "manager":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Manager role required")
    return user

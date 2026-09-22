"""
Test env setup — must run before any project module is imported, since
config.py reads these at import time. conftest.py is collected by pytest
before test modules in the same directory, which is what makes that ordering
hold here.

- DATABASE_URL points at a throwaway temp sqlite file (never the real
  data/policybot.db), fresh per test session.
- GROQ_API_KEY is a dummy value: enough for ChatGroq's constructor to not
  raise (it validates that *something* is set, see langchain_groq's
  validate_environment) without making a real network call — tests never
  call .invoke() on the LLM.
"""
import os
import tempfile

_tmp_db_fd, _tmp_db_path = tempfile.mkstemp(suffix=".db")
os.close(_tmp_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db_path}"
os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from db.session import Base, engine  # noqa: E402
from db import models  # noqa: E402,F401 (registers models with Base before create_all)


@pytest.fixture(scope="session", autouse=True)
def _init_db():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture()
def client():
    from api.app import app
    return TestClient(app)


@pytest.fixture()
def token_for():
    from auth.security import create_access_token

    def _make(username: str, role: str) -> str:
        return create_access_token(subject=username, role=role)

    return _make


@pytest.fixture()
def auth_headers(token_for):
    def _make(username: str, role: str) -> dict:
        return {"Authorization": f"Bearer {token_for(username, role)}"}

    return _make

# scripts/db_init.py
from db.session import Base, engine
from db.models import LeaveRequest, AuditLog  # noqa: F401 (imported so create_all sees them)

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("DB ready: data/policybot.db")

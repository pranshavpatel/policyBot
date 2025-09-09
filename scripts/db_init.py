# scripts/db_init.py
from db.session import Base, engine
from db.models import LeaveRequest

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("DB ready: data/policybot.db")

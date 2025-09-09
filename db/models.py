# db/models.py
from sqlalchemy import Column, String, Date, DateTime
from datetime import datetime
from db.session import Base

class LeaveRequest(Base):
    __tablename__ = "leave_requests"
    id = Column(String, primary_key=True)                 # uuid
    user = Column(String, index=True, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    reason = Column(String, nullable=False)
    status = Column(String, index=True, nullable=False)   # submitted|approved|rejected|cancelled
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime)

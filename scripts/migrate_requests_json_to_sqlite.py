# scripts/migrate_requests_json_to_sqlite.py
import json, os
from datetime import datetime
from db.session import SessionLocal
from db.models import LeaveRequest

SRC = "data/requests_db.json"

def main():
    if not os.path.exists(SRC):
        print("No JSON file found; nothing to migrate.")
        return
    data = json.load(open(SRC, "r", encoding="utf-8"))
    with SessionLocal() as s:
        for r in data:
            if s.get(LeaveRequest, r["id"]): 
                continue
            obj = LeaveRequest(
                id=r["id"], user=r["user"],
                start_date=datetime.fromisoformat(r["start_date"]).date(),
                end_date=datetime.fromisoformat(r["end_date"]).date(),
                reason=r["reason"], status=r.get("status","submitted"),
                created_at=datetime.fromisoformat(r["created_at"].replace("Z","")),
                updated_at=datetime.fromisoformat(r["updated_at"].replace("Z","")) if r.get("updated_at") else None
            )
            s.add(obj)
        s.commit()
    print("Migration complete.")

if __name__ == "__main__":
    main()

# tools/holiday_check.py
"""
Holiday Check tool (simulated).
- check_holiday(date_str) -> {"is_holiday": bool, "name": str | None, "date": "YYYY-MM-DD"}
- list_holidays(year=None) -> [{"date": "YYYY-MM-DD", "name": "..."}]
- next_holidays(n=5, start_date=None) -> next N holidays after today/start_date

Date format: ISO 'YYYY-MM-DD'
Edit the HOLIDAYS dict to fit your company calendar.
"""

from __future__ import annotations
from datetime import datetime, date
from typing import List, Dict, Any, Optional

# ----- Company holiday calendar (fixed-date examples). Add/remove as needed.
# Format: "MM-DD": "Holiday Name"
FIXED_HOLIDAYS = {
    "01-01": "New Year’s Day",
    "01-15": "Founders Day",           # example internal holiday
    "07-04": "Independence Day",
    "09-01": "Company Day",            # example internal holiday
    "11-28": "Thanksgiving Day",       # for demo: fixed date; real rule is 4th Thu in Nov
    "12-25": "Christmas Day",
}

def _parse_iso_date(s: str) -> date:
    try:
        return datetime.fromisoformat(s).date()
    except Exception as e:
        raise ValueError("Invalid date format. Use ISO 'YYYY-MM-DD'.") from e

def _format(d: date) -> str:
    return d.isoformat()

def _mmdd(d: date) -> str:
    return d.strftime("%m-%d")

def check_holiday(date_str: str) -> Dict[str, Any]:
    """
    Returns whether the given date is a company holiday.
    Output:
      {"is_holiday": bool, "name": str | None, "date": "YYYY-MM-DD"}
    """
    d = _parse_iso_date(date_str)
    name = FIXED_HOLIDAYS.get(_mmdd(d))
    return {"is_holiday": name is not None, "name": name, "date": _format(d)}

def list_holidays(year: Optional[int] = None) -> List[Dict[str, str]]:
    """
    List holidays for a given year (defaults to current year).
    Output: [{"date":"YYYY-MM-DD","name":"..."}]
    """
    y = year or date.today().year
    out = []
    for mmdd, name in FIXED_HOLIDAYS.items():
        m, d = map(int, mmdd.split("-"))
        out.append({"date": f"{y:04d}-{m:02d}-{d:02d}", "name": name})
    # Sort chronologically
    out.sort(key=lambda x: x["date"])
    return out

def next_holidays(n: int = 5, start_date: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Next N holidays after today or after start_date (inclusive if same day).
    """
    start = _parse_iso_date(start_date) if start_date else date.today()
    year_order = [start.year, start.year + 1]  # simple lookahead across year boundary
    pool = []
    for y in year_order:
        pool.extend(list_holidays(y))

    # keep those >= start
    pool = [h for h in pool if _parse_iso_date(h["date"]) >= start]
    return pool[: max(0, n)]

# Optional: simple JSON schema for planners
CHECK_HOLIDAY_SCHEMA: Dict[str, Any] = {
    "name": "check_holiday",
    "description": "Check if a date is a company holiday.",
    "schema": {
        "type": "object",
        "properties": {"date_str": {"type": "string", "description": "ISO date YYYY-MM-DD"}},
        "required": ["date_str"],
        "additionalProperties": False,
    },
}

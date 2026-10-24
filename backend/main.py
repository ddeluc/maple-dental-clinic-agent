"""Fake practice management system: a small FastAPI app over clinic.db.

Run with: uv run uvicorn backend.main:app --reload
Interactive docs: http://localhost:8000/docs
"""

import sqlite3
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel

from backend.seed import DB_PATH
from backend.verification import normalize_name, normalize_phone

app = FastAPI(title="Maple Dental practice management (fake)")

PATIENT_NOT_FOUND = "No patient matches that name, date of birth and phone number."
PATIENT_AMBIGUOUS = (
    "More than one patient record matches that name, date of birth and phone"
    " number, so the patient can't be identified."
)

# A slot is open when no booked appointment points to it (cancelled ones
# don't count). Times are ISO 8601 text, which sorts and compares correctly
# as plain strings, so the date range is a simple text comparison.
OPEN_SLOTS_QUERY = """
    SELECT
        slots.id,
        slots.location_id,
        slots.provider_id,
        providers.name AS provider_name,
        slots.start_time,
        slots.end_time
    FROM slots
    JOIN providers ON providers.id = slots.provider_id
    WHERE slots.location_id = ?
      AND providers.role = ?
      AND slots.start_time >= ?
      AND slots.start_time < ?
      AND NOT EXISTS (
          SELECT 1 FROM appointments
          WHERE appointments.slot_id = slots.id
            AND appointments.status = 'booked'
      )
    ORDER BY slots.start_time, providers.name
"""


class Location(BaseModel):
    id: int
    name: str


class Slot(BaseModel):
    id: int
    location_id: int
    provider_id: int
    provider_name: str
    start_time: str
    end_time: str


class VerifyRequest(BaseModel):
    name: str
    date_of_birth: date
    phone: str


class VerifyResponse(BaseModel):
    patient_id: int


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection whose rows can be read by column name."""
    # FastAPI can run a dependency and its endpoint on different worker
    # threads, which sqlite3 refuses by default. Turning the check off is
    # safe here because each connection is used by one request only.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_db() -> Iterator[sqlite3.Connection]:
    """Give each request its own connection and close it afterwards."""
    conn = connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


Database = Annotated[sqlite3.Connection, Depends(get_db)]


@app.get("/locations/{location_id}")
def get_location(location_id: int, db: Database) -> Location:
    row = db.execute(
        "SELECT id, name FROM locations WHERE id = ?", (location_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No location with id {location_id}.")
    return Location(id=row["id"], name=row["name"])


@app.get("/slots")
def get_open_slots(
    db: Database,
    location: Annotated[int, Query(description="Location id")],
    # "type" and "from" are the public parameter names, but "from" is a
    # Python keyword and "type" is a built-in, so the Python names differ
    # and alias= maps the URL names onto them.
    appointment_type: Annotated[str, Query(alias="type", description="cleaning or checkup")],
    first_day: Annotated[date, Query(alias="from", description="YYYY-MM-DD")],
    last_day: Annotated[date, Query(alias="to", description="YYYY-MM-DD, inclusive")],
) -> list[Slot]:
    """Return the open slots for one location and appointment type."""
    if first_day > last_day:
        raise HTTPException(status_code=422, detail="'from' must be on or before 'to'.")

    location_row = db.execute(
        "SELECT id FROM locations WHERE id = ?", (location,)
    ).fetchone()
    if location_row is None:
        raise HTTPException(status_code=404, detail=f"No location with id {location}.")

    # The appointment type decides which kind of provider's slots to show:
    # cleanings are with hygienists, checkups with dentists.
    type_row = db.execute(
        "SELECT provider_role FROM appointment_types WHERE name = ?",
        (appointment_type,),
    ).fetchone()
    if type_row is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown appointment type '{appointment_type}'."
        )

    # Comparing against the start of the day after last_day includes every
    # slot on last_day itself: "2026-10-30T16:00" < "2026-10-31".
    day_after_last = last_day + timedelta(days=1)
    rows = db.execute(
        OPEN_SLOTS_QUERY,
        (
            location,
            type_row["provider_role"],
            first_day.isoformat(),
            day_after_last.isoformat(),
        ),
    ).fetchall()

    slots = []
    for row in rows:
        slots.append(
            Slot(
                id=row["id"],
                location_id=row["location_id"],
                provider_id=row["provider_id"],
                provider_name=row["provider_name"],
                start_time=row["start_time"],
                end_time=row["end_time"],
            )
        )
    return slots


@app.post("/patients/verify")
def verify_patient(request: VerifyRequest, db: Database) -> VerifyResponse:
    """Find the one patient whose name, date of birth and phone all match."""
    # SQLite can't compare text while ignoring accents, so SQL only narrows
    # the search to the exact date of birth, and Python compares the
    # normalized names and phone numbers of those few patients.
    rows = db.execute(
        "SELECT id, first_name, last_name, phone FROM patients WHERE date_of_birth = ?",
        (request.date_of_birth.isoformat(),),
    ).fetchall()

    wanted_name = normalize_name(request.name)
    wanted_phone = normalize_phone(request.phone)

    matching_ids = []
    for row in rows:
        full_name = f"{row['first_name']} {row['last_name']}"
        name_matches = normalize_name(full_name) == wanted_name
        phone_matches = normalize_phone(row["phone"]) == wanted_phone
        if name_matches and phone_matches:
            matching_ids.append(row["id"])

    # The error never says which detail was wrong, so a caller can't use it
    # to guess a patient's details one field at a time.
    if not matching_ids:
        raise HTTPException(status_code=404, detail=PATIENT_NOT_FOUND)
    # Two records matching all three details means we can't tell which one
    # is the caller, and acting on the wrong record is worse than refusing.
    if len(matching_ids) > 1:
        raise HTTPException(status_code=409, detail=PATIENT_AMBIGUOUS)
    return VerifyResponse(patient_id=matching_ids[0])

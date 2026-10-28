"""Fake practice management system: a small FastAPI app over clinic.db.

Run with: uv run uvicorn backend.main:app --reload
Interactive docs: http://localhost:8000/docs
"""

import sqlite3
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel

from backend.clinic_facts import CLOSED, DAY_NAMES, load_clinic_facts
from backend.seed import DB_PATH
from backend.verification import normalize_name, normalize_phone

app = FastAPI(title="Maple Dental practice management (fake)")

PATIENT_NOT_FOUND = "No patient matches that name, date of birth and phone number."
PATIENT_AMBIGUOUS = (
    "More than one patient record matches that name, date of birth and phone"
    " number, so the patient can't be identified."
)
SLOT_TAKEN = "That slot is already booked."

# Read once at startup. uvicorn --reload only watches .py files, so restart
# the server after editing config/maple_dental.yaml.
CLINIC_FACTS = load_clinic_facts()
# Insurers are the same at every location; the other topics differ by location.
ClinicTopic = Literal["hours", "address", "parking", "insurers"]

# The project definition allows bookings from now up to 4 weeks ahead.
BOOKING_WINDOW_DAYS = 28

# Everything the API returns about one appointment, plus the provider role
# its type needs, which rescheduling uses to check the new slot.
APPOINTMENT_QUERY = """
    SELECT
        appointments.id,
        appointments.patient_id,
        appointments.slot_id,
        appointment_types.name AS type,
        appointment_types.provider_role,
        appointments.status,
        appointments.cancel_reason,
        slots.location_id,
        providers.name AS provider_name,
        slots.start_time,
        slots.end_time
    FROM appointments
    JOIN appointment_types ON appointment_types.id = appointments.appointment_type_id
    JOIN slots ON slots.id = appointments.slot_id
    JOIN providers ON providers.id = slots.provider_id
    WHERE appointments.id = ?
"""

# A slot is open when no booked appointment points to it (cancelled ones
# don't count). Times are ISO 8601 text, which sorts and compares correctly
# as plain strings, so the date range is a simple text comparison. The last
# condition drops slots that started before "now".
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
      AND slots.start_time >= ?
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


class ClinicInfo(BaseModel):
    topic: str
    location: str | None  # None for insurers, which are clinic-wide
    # Hours are {"monday": "08:00-17:00", ...}, address and parking are
    # text, and insurers are a list of names.
    details: dict[str, str] | str | list[str]
    # Hours only: for each day this location is closed, the other locations
    # open that day, e.g. {"saturday": {"Plateau": "09:00-13:00"}}.
    open_elsewhere: dict[str, dict[str, str]] | None = None


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


class BookRequest(BaseModel):
    patient_id: int
    slot_id: int
    type: str  # "cleaning" or "checkup"


class RescheduleRequest(BaseModel):
    slot_id: int


class Appointment(BaseModel):
    id: int
    patient_id: int
    slot_id: int
    type: str
    status: str
    cancel_reason: str | None
    location_id: int
    provider_name: str
    start_time: str
    end_time: str


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


def get_now() -> datetime:
    """Return the current time, in the same clinic-local form as slot times."""
    # A dependency instead of calling datetime.now() inside the endpoints,
    # so tests can pin "now" and check the booking window on fixed seed data.
    # This uses the server's clock, which must be set to Montreal time.
    return datetime.now()


Now = Annotated[datetime, Depends(get_now)]


@app.get("/locations")
def list_locations(db: Database) -> list[Location]:
    rows = db.execute("SELECT id, name FROM locations ORDER BY id").fetchall()
    locations = []
    for row in rows:
        locations.append(Location(id=row["id"], name=row["name"]))
    return locations


@app.get("/locations/{location_id}")
def get_location(location_id: int, db: Database) -> Location:
    row = db.execute(
        "SELECT id, name FROM locations WHERE id = ?", (location_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No location with id {location_id}.")
    return Location(id=row["id"], name=row["name"])


def find_open_elsewhere(location_name: str) -> dict[str, dict[str, str]]:
    """For each day a location is closed, list the other locations open then."""
    # Telling the model where else it can send the caller works far more
    # reliably than asking it, in the prompt, to look up every other
    # location's hours by itself.
    hours = CLINIC_FACTS["locations"][location_name]["hours"]
    open_elsewhere = {}
    for day_name in DAY_NAMES:
        if hours[day_name] != CLOSED:
            continue
        open_locations = {}
        for other_name, other_facts in CLINIC_FACTS["locations"].items():
            other_hours = other_facts["hours"][day_name]
            if other_name != location_name and other_hours != CLOSED:
                open_locations[other_name] = other_hours
        open_elsewhere[day_name] = open_locations
    return open_elsewhere


@app.get("/clinic-info")
def get_clinic_info(
    topic: Annotated[ClinicTopic, Query(description="hours, address, parking or insurers")],
    location: Annotated[
        str | None, Query(description="Location name; not needed for insurers")
    ] = None,
) -> ClinicInfo:
    """Return one clinic fact from config/maple_dental.yaml."""
    if topic == "insurers":
        return ClinicInfo(topic=topic, location=None, details=CLINIC_FACTS["insurers"])

    location_names = ", ".join(CLINIC_FACTS["locations"])
    if location is None:
        raise HTTPException(
            status_code=422,
            detail=f"The {topic} answer depends on the location. Choose one of: {location_names}.",
        )
    location_facts = CLINIC_FACTS["locations"].get(location)
    if location_facts is None:
        raise HTTPException(
            status_code=404,
            detail=f"No location named '{location}'. The locations are: {location_names}.",
        )
    if topic == "hours":
        return ClinicInfo(
            topic=topic,
            location=location,
            details=location_facts["hours"],
            open_elsewhere=find_open_elsewhere(location),
        )
    return ClinicInfo(topic=topic, location=location, details=location_facts[topic])


@app.get("/slots")
def get_open_slots(
    db: Database,
    now: Now,
    location: Annotated[int, Query(description="Location id")],
    # "type" and "from" are the public parameter names, but "from" is a
    # Python keyword and "type" is a built-in, so the Python names differ
    # and alias= maps the URL names onto them.
    appointment_type: Annotated[str, Query(alias="type", description="cleaning or checkup")],
    first_day: Annotated[date, Query(alias="from", description="YYYY-MM-DD")],
    last_day: Annotated[date, Query(alias="to", description="YYYY-MM-DD, inclusive")],
) -> list[Slot]:
    """Return the open slots for one location and appointment type.

    Slots that have already started are left out, even inside the requested
    dates, because they can no longer be booked.
    """
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
            # Same "2026-10-26T09:00" form as start_time, so the text
            # comparison works.
            now.isoformat(timespec="minutes"),
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


# --- Appointments: booking, rescheduling, cancelling ---


def find_appointment_row(db: sqlite3.Connection, appointment_id: int) -> sqlite3.Row:
    row = db.execute(APPOINTMENT_QUERY, (appointment_id,)).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"No appointment with id {appointment_id}."
        )
    return row


def appointment_from_row(row: sqlite3.Row) -> Appointment:
    return Appointment(
        id=row["id"],
        patient_id=row["patient_id"],
        slot_id=row["slot_id"],
        type=row["type"],
        status=row["status"],
        cancel_reason=row["cancel_reason"],
        location_id=row["location_id"],
        provider_name=row["provider_name"],
        start_time=row["start_time"],
        end_time=row["end_time"],
    )


def find_slot_row(db: sqlite3.Connection, slot_id: int) -> sqlite3.Row:
    row = db.execute(
        "SELECT slots.id, slots.start_time, providers.role AS provider_role"
        " FROM slots JOIN providers ON providers.id = slots.provider_id"
        " WHERE slots.id = ?",
        (slot_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No slot with id {slot_id}.")
    return row


def check_booking_window(start_time: str, now: datetime) -> None:
    """Refuse slots in the past or more than BOOKING_WINDOW_DAYS ahead."""
    start = datetime.fromisoformat(start_time)
    if start < now:
        raise HTTPException(status_code=422, detail="That slot is in the past.")
    latest_start = now + timedelta(days=BOOKING_WINDOW_DAYS)
    if start > latest_start:
        raise HTTPException(
            status_code=422,
            detail=f"Appointments can be booked at most {BOOKING_WINDOW_DAYS} days ahead.",
        )


def check_slot_is_open(db: sqlite3.Connection, slot_id: int) -> None:
    row = db.execute(
        "SELECT 1 FROM appointments WHERE slot_id = ? AND status = 'booked'",
        (slot_id,),
    ).fetchone()
    if row is not None:
        raise HTTPException(status_code=409, detail=SLOT_TAKEN)


def check_slot_can_be_booked(
    db: sqlite3.Connection, slot: sqlite3.Row, provider_role: str, now: datetime
) -> None:
    """Run every check a slot must pass before an appointment can use it."""
    # Cleanings are with hygienists and checkups with dentists, so a
    # checkup can't be put in a hygienist's slot.
    if slot["provider_role"] != provider_role:
        raise HTTPException(
            status_code=422,
            detail=f"Slot {slot['id']} is with a {slot['provider_role']},"
            f" but this appointment needs a {provider_role}.",
        )
    check_booking_window(slot["start_time"], now)
    check_slot_is_open(db, slot["id"])


def write_slot_change(db: sqlite3.Connection, sql: str, parameters: tuple) -> sqlite3.Cursor:
    """Run an insert or update that puts a booked appointment in a slot."""
    # check_slot_is_open runs first, but two requests can both pass it before
    # either one writes. The one_booking_per_slot index then rejects the
    # second write, and that is reported the same way as the normal check.
    try:
        cursor = db.execute(sql, parameters)
        db.commit()
    except sqlite3.IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail=SLOT_TAKEN)
    return cursor


@app.post("/appointments", status_code=201)
def book_appointment(request: BookRequest, db: Database, now: Now) -> Appointment:
    """Book an open slot for a patient."""
    patient_row = db.execute(
        "SELECT id FROM patients WHERE id = ?", (request.patient_id,)
    ).fetchone()
    if patient_row is None:
        raise HTTPException(
            status_code=404, detail=f"No patient with id {request.patient_id}."
        )

    type_row = db.execute(
        "SELECT id, provider_role FROM appointment_types WHERE name = ?",
        (request.type,),
    ).fetchone()
    if type_row is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown appointment type '{request.type}'."
        )

    slot = find_slot_row(db, request.slot_id)
    check_slot_can_be_booked(db, slot, type_row["provider_role"], now)

    cursor = write_slot_change(
        db,
        "INSERT INTO appointments"
        " (patient_id, slot_id, appointment_type_id, status, created_at)"
        " VALUES (?, ?, ?, 'booked', ?)",
        (
            request.patient_id,
            request.slot_id,
            type_row["id"],
            now.isoformat(timespec="seconds"),
        ),
    )
    return appointment_from_row(find_appointment_row(db, cursor.lastrowid))


@app.patch("/appointments/{appointment_id}")
def reschedule_appointment(
    appointment_id: int, request: RescheduleRequest, db: Database, now: Now
) -> Appointment:
    """Move a booked appointment to another open slot."""
    appointment = find_appointment_row(db, appointment_id)
    if appointment["status"] != "booked":
        raise HTTPException(
            status_code=409, detail="A cancelled appointment can't be rescheduled."
        )

    new_slot = find_slot_row(db, request.slot_id)
    check_slot_can_be_booked(db, new_slot, appointment["provider_role"], now)

    # Changing slot_id is all it takes: the old slot has no booked
    # appointment pointing to it any more, so GET /slots shows it as open.
    write_slot_change(
        db,
        "UPDATE appointments SET slot_id = ? WHERE id = ?",
        (request.slot_id, appointment_id),
    )
    return appointment_from_row(find_appointment_row(db, appointment_id))


@app.delete("/appointments/{appointment_id}")
def cancel_appointment(
    appointment_id: int,
    db: Database,
    reason: Annotated[str, Query(min_length=1, description="Why the patient cancelled")],
) -> Appointment:
    """Cancel an appointment and record why, which frees its slot."""
    appointment = find_appointment_row(db, appointment_id)
    if appointment["status"] != "booked":
        raise HTTPException(
            status_code=409, detail="That appointment is already cancelled."
        )

    # The row is kept and marked cancelled rather than deleted, so the
    # clinic keeps a history of cancellations and their reasons.
    db.execute(
        "UPDATE appointments SET status = 'cancelled', cancel_reason = ? WHERE id = ?",
        (reason, appointment_id),
    )
    db.commit()
    return appointment_from_row(find_appointment_row(db, appointment_id))

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from backend.seed import SATURDAY, SLOT_DAYS, opening_hours, seed

# A fixed Monday keeps these tests independent of the day they run.
FIRST_DAY = date(2026, 10, 26)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "clinic.db"
    seed(path, first_day=FIRST_DAY)
    return path


@pytest.fixture
def conn(db_path: Path):
    connection = sqlite3.connect(db_path)
    yield connection
    connection.close()


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_every_table_has_rows(conn):
    tables = [
        "locations", "providers", "patients",
        "appointment_types", "slots", "appointments",
    ]
    for table in tables:
        assert count_rows(conn, table) > 0, f"{table} is empty"


def test_clinic_matches_project_definition(conn):
    assert count_rows(conn, "locations") == 3
    assert count_rows(conn, "patients") == 50

    roles = conn.execute(
        "SELECT role, COUNT(*) FROM providers GROUP BY role"
    ).fetchall()
    assert dict(roles) == {"dentist": 6, "hygienist": 5}

    types = conn.execute(
        "SELECT name, provider_role, duration_minutes FROM appointment_types"
    ).fetchall()
    assert set(types) == {("cleaning", "hygienist", 60), ("checkup", "dentist", 30)}


def test_seeding_twice_rebuilds_from_scratch(db_path):
    # Add a stray row, then seed again: it must be gone and counts unchanged.
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO locations (name) VALUES ('Stray location')")
    conn.commit()
    conn.close()

    seed(db_path, first_day=FIRST_DAY)

    conn = sqlite3.connect(db_path)
    assert count_rows(conn, "locations") == 3
    assert count_rows(conn, "patients") == 50
    conn.close()


def slots_with_location(conn: sqlite3.Connection) -> list[tuple[str, datetime, datetime]]:
    rows = conn.execute(
        "SELECT locations.name, slots.start_time, slots.end_time FROM slots"
        " JOIN locations ON locations.id = slots.location_id"
    ).fetchall()
    slots = []
    for location_name, start_text, end_text in rows:
        slots.append(
            (location_name, datetime.fromisoformat(start_text), datetime.fromisoformat(end_text))
        )
    return slots


def test_saturday_slots_only_at_plateau(conn):
    saturday_locations = set()
    for location_name, start, _end in slots_with_location(conn):
        if start.weekday() == SATURDAY:
            saturday_locations.add(location_name)
    assert saturday_locations == {"Plateau"}


def test_slots_fall_within_opening_hours_and_four_weeks(conn):
    last_day = FIRST_DAY + timedelta(days=SLOT_DAYS - 1)
    for location_name, start, end in slots_with_location(conn):
        assert FIRST_DAY <= start.date() <= last_day
        hours = opening_hours(location_name, start.date())
        assert hours is not None, f"{location_name} slot on a closed day: {start}"
        opens, closes = hours
        assert opens <= start.time()
        assert end.time() <= closes


def test_slot_length_matches_provider_role(conn):
    rows = conn.execute(
        "SELECT providers.role, slots.start_time, slots.end_time FROM slots"
        " JOIN providers ON providers.id = slots.provider_id"
    ).fetchall()
    expected_minutes = {"dentist": 30, "hygienist": 60}
    for role, start_text, end_text in rows:
        length = datetime.fromisoformat(end_text) - datetime.fromisoformat(start_text)
        assert length == timedelta(minutes=expected_minutes[role])


def test_database_rejects_double_booking(conn):
    slot_id, patient_id, type_id = conn.execute(
        "SELECT slot_id, patient_id, appointment_type_id FROM appointments"
        " WHERE status = 'booked' LIMIT 1"
    ).fetchone()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO appointments"
            " (patient_id, slot_id, appointment_type_id, status, created_at)"
            " VALUES (?, ?, ?, 'booked', '2026-10-26T10:00:00')",
            (patient_id, slot_id, type_id),
        )

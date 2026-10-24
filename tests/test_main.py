import sqlite3
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import PATIENT_AMBIGUOUS, PATIENT_NOT_FOUND, app, connect, get_db
from backend.seed import seed

# A fixed Monday keeps these tests independent of the day they run.
FIRST_DAY = date(2026, 10, 26)
PLATEAU_ID = 1  # locations are inserted in order: Plateau, Verdun, Saint-Laurent


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "clinic.db"
    seed(path, first_day=FIRST_DAY)
    return path


@pytest.fixture
def client(db_path: Path) -> Iterator[TestClient]:
    # Point the app at the temporary database instead of backend/clinic.db.
    def get_test_db() -> Iterator[sqlite3.Connection]:
        conn = connect(db_path)
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[get_db] = get_test_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def patient_id_with_phone(db_path: Path, phone: str) -> int:
    """Look up a seeded patient's id by the phone number exactly as stored."""
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT id FROM patients WHERE phone = ?", (phone,)).fetchone()
    conn.close()
    return row[0]


def verify(client: TestClient, name: str, date_of_birth: str, phone: str):
    return client.post(
        "/patients/verify",
        json={"name": name, "date_of_birth": date_of_birth, "phone": phone},
    )


# --- GET /locations/{id} ---


def test_get_location_returns_its_name(client):
    response = client.get(f"/locations/{PLATEAU_ID}")
    assert response.status_code == 200
    assert response.json() == {"id": PLATEAU_ID, "name": "Plateau"}


def test_get_unknown_location_returns_404(client):
    response = client.get("/locations/999")
    assert response.status_code == 404


# --- GET /slots ---


def get_slots(client: TestClient, location: int, type_name: str, first: date, last: date):
    return client.get(
        "/slots",
        params={
            "location": location,
            "type": type_name,
            "from": first.isoformat(),
            "to": last.isoformat(),
        },
    )


def provider_roles(db_path: Path) -> dict[int, str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, role FROM providers").fetchall()
    conn.close()
    return dict(rows)


def test_slots_match_location_type_and_dates(client, db_path):
    last_day = FIRST_DAY + timedelta(days=2)
    response = get_slots(client, PLATEAU_ID, "checkup", FIRST_DAY, last_day)
    assert response.status_code == 200
    slots = response.json()
    assert len(slots) > 0

    roles = provider_roles(db_path)
    for slot in slots:
        assert slot["location_id"] == PLATEAU_ID
        assert roles[slot["provider_id"]] == "dentist"
        start_day = date.fromisoformat(slot["start_time"][:10])
        assert FIRST_DAY <= start_day <= last_day

    start_times = []
    for slot in slots:
        start_times.append(slot["start_time"])
    assert start_times == sorted(start_times)


def test_slots_include_the_whole_last_day(client):
    response = get_slots(client, PLATEAU_ID, "cleaning", FIRST_DAY, FIRST_DAY)
    slots = response.json()
    assert len(slots) > 0
    for slot in slots:
        assert slot["start_time"].startswith(FIRST_DAY.isoformat())
    # The Plateau closes at 17:00 and cleanings are 60 minutes long.
    assert slots[-1]["start_time"] == f"{FIRST_DAY.isoformat()}T16:00"


def test_booked_slots_are_not_returned(client, db_path):
    conn = sqlite3.connect(db_path)
    slot_id, location_id, type_name, start_time = conn.execute(
        "SELECT slots.id, slots.location_id, appointment_types.name, slots.start_time"
        " FROM appointments"
        " JOIN slots ON slots.id = appointments.slot_id"
        " JOIN appointment_types ON appointment_types.id = appointments.appointment_type_id"
        " WHERE appointments.status = 'booked' LIMIT 1"
    ).fetchone()
    conn.close()

    day = date.fromisoformat(start_time[:10])
    response = get_slots(client, location_id, type_name, day, day)
    returned_ids = []
    for slot in response.json():
        returned_ids.append(slot["id"])
    assert len(returned_ids) > 0
    assert slot_id not in returned_ids


def test_slots_with_cancelled_appointment_are_returned(client, db_path):
    conn = sqlite3.connect(db_path)
    slot_id, location_id, type_name, start_time = conn.execute(
        "SELECT slots.id, slots.location_id, appointment_types.name, slots.start_time"
        " FROM appointments"
        " JOIN slots ON slots.id = appointments.slot_id"
        " JOIN appointment_types ON appointment_types.id = appointments.appointment_type_id"
        " WHERE appointments.status = 'cancelled' LIMIT 1"
    ).fetchone()
    conn.close()

    day = date.fromisoformat(start_time[:10])
    response = get_slots(client, location_id, type_name, day, day)
    returned_ids = []
    for slot in response.json():
        returned_ids.append(slot["id"])
    assert slot_id in returned_ids


def test_slots_unknown_location_returns_404(client):
    response = get_slots(client, 999, "cleaning", FIRST_DAY, FIRST_DAY)
    assert response.status_code == 404


def test_slots_unknown_type_returns_404(client):
    response = get_slots(client, PLATEAU_ID, "whitening", FIRST_DAY, FIRST_DAY)
    assert response.status_code == 404


def test_slots_from_after_to_returns_422(client):
    response = get_slots(client, PLATEAU_ID, "cleaning", FIRST_DAY, FIRST_DAY - timedelta(days=1))
    assert response.status_code == 422


# --- POST /patients/verify ---


def test_verify_correct_details_returns_patient_id(client, db_path):
    response = verify(client, "Marie Tremblay", "1985-03-04", "514-555-0100")
    assert response.status_code == 200
    assert response.json() == {"patient_id": patient_id_with_phone(db_path, "514-555-0100")}


def test_verify_ignores_accents_case_and_phone_format(client, db_path):
    # The exact example from the step's "Done when" list.
    response = verify(client, "helene tremblay", "1979-02-14", "(514) 555-0142")
    assert response.status_code == 200
    assert response.json() == {"patient_id": patient_id_with_phone(db_path, "514-555-0142")}


def test_verify_matches_stored_country_code(client, db_path):
    response = verify(client, "Jean-Francois Cote-Lavoie", "1990-11-23", "514-555-0104")
    assert response.status_code == 200
    assert response.json() == {"patient_id": patient_id_with_phone(db_path, "+1 514 555 0104")}


def test_verify_wrong_phone_returns_not_found(client):
    response = verify(client, "Hélène Tremblay", "1979-02-14", "514-555-0199")
    assert response.status_code == 404
    assert response.json() == {"detail": PATIENT_NOT_FOUND}


def test_verify_wrong_name_returns_not_found(client):
    response = verify(client, "Marie Tremblay", "1979-02-14", "514-555-0142")
    assert response.status_code == 404
    assert response.json() == {"detail": PATIENT_NOT_FOUND}


def test_verify_swapped_day_and_month_returns_not_found(client):
    # 1985-04-03 belongs to the other Marie Tremblay, who has a different phone.
    response = verify(client, "Marie Tremblay", "1985-04-03", "514-555-0100")
    assert response.status_code == 404


def test_verify_phone_tells_same_name_and_birthday_apart(client, db_path):
    response = verify(client, "Michael Smith", "1968-01-30", "438-555-0107")
    assert response.status_code == 200
    assert response.json() == {"patient_id": patient_id_with_phone(db_path, "438-555-0107")}


def test_verify_two_identical_records_returns_409(client, db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO patients (first_name, last_name, date_of_birth, phone)"
        " VALUES ('Helene', 'Tremblay', '1979-02-14', '514 555 0142')"
    )
    conn.commit()
    conn.close()

    response = verify(client, "Hélène Tremblay", "1979-02-14", "514-555-0142")
    assert response.status_code == 409
    assert response.json() == {"detail": PATIENT_AMBIGUOUS}


def test_verify_invalid_date_of_birth_returns_422(client):
    response = verify(client, "Hélène Tremblay", "14/02/1979", "514-555-0142")
    assert response.status_code == 422

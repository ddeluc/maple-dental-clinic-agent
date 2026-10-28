import sqlite3
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import (
    BOOKING_WINDOW_DAYS,
    PATIENT_AMBIGUOUS,
    PATIENT_NOT_FOUND,
    SLOT_TAKEN,
    app,
    connect,
    get_db,
    get_now,
)
from backend.seed import seed

# A fixed Monday keeps these tests independent of the day they run.
FIRST_DAY = date(2026, 10, 26)
# An hour before the first slots open, so every seeded slot is bookable.
TEST_NOW = datetime(2026, 10, 26, 7, 0)
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
    set_now(TEST_NOW)
    yield TestClient(app)
    app.dependency_overrides.clear()


def set_now(moment: datetime) -> None:
    """Make the app believe the current time is `moment`."""
    # The lambda takes no arguments, so FastAPI calls it like get_now and
    # gets back the moment captured here.
    app.dependency_overrides[get_now] = lambda: moment


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


# --- GET /locations ---


def test_list_locations_returns_all_three_in_id_order(client):
    response = client.get("/locations")
    assert response.status_code == 200
    assert response.json() == [
        {"id": 1, "name": "Plateau"},
        {"id": 2, "name": "Verdun"},
        {"id": 3, "name": "Saint-Laurent"},
    ]


# --- GET /clinic-info ---


def test_clinic_info_hours_for_one_location(client):
    response = client.get("/clinic-info", params={"topic": "hours", "location": "Verdun"})
    assert response.status_code == 200
    body = response.json()
    assert body["location"] == "Verdun"
    assert body["details"]["saturday"] == "closed"
    assert body["details"]["friday"] == "08:00-17:00"


def test_clinic_info_hours_say_where_else_is_open_on_closed_days(client):
    response = client.get("/clinic-info", params={"topic": "hours", "location": "Verdun"})
    open_elsewhere = response.json()["open_elsewhere"]
    # Only Verdun's closed days are listed, and nothing opens on Sunday.
    assert open_elsewhere == {"saturday": {"Plateau": "09:00-13:00"}, "sunday": {}}


def test_clinic_info_address_has_no_open_elsewhere(client):
    response = client.get("/clinic-info", params={"topic": "address", "location": "Verdun"})
    assert response.json()["open_elsewhere"] is None


def test_clinic_info_address_is_text(client):
    response = client.get("/clinic-info", params={"topic": "address", "location": "Plateau"})
    assert "rue Saint-Denis" in response.json()["details"]


def test_clinic_info_insurers_ignore_location(client):
    response = client.get("/clinic-info", params={"topic": "insurers"})
    assert response.status_code == 200
    body = response.json()
    assert body["location"] is None
    assert "Desjardins Insurance" in body["details"]


def test_clinic_info_location_topic_without_location_returns_422(client):
    response = client.get("/clinic-info", params={"topic": "parking"})
    assert response.status_code == 422
    assert "Plateau, Verdun, Saint-Laurent" in response.json()["detail"]


def test_clinic_info_unknown_location_returns_404(client):
    response = client.get("/clinic-info", params={"topic": "hours", "location": "Laval"})
    assert response.status_code == 404


def test_clinic_info_unknown_topic_returns_422(client):
    response = client.get("/clinic-info", params={"topic": "dress code", "location": "Verdun"})
    assert response.status_code == 422


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


def test_slots_that_already_started_are_not_returned(client):
    set_now(datetime(2026, 10, 26, 12, 30))  # lunchtime on FIRST_DAY
    response = get_slots(client, PLATEAU_ID, "cleaning", FIRST_DAY, FIRST_DAY)
    start_times = []
    for slot in response.json():
        start_times.append(slot["start_time"])
    # The 12:00 cleaning started half an hour ago; 13:00 is the next one.
    assert start_times[0] == f"{FIRST_DAY.isoformat()}T13:00"


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


# --- Appointments ---


MARIE_PHONE = "514-555-0100"  # a seeded patient with a unique record
HELENE_PHONE = "514-555-0142"  # a second patient, for "someone else" cases


def book(client: TestClient, patient_id: int, slot_id: int, type_name: str):
    return client.post(
        "/appointments",
        json={"patient_id": patient_id, "slot_id": slot_id, "type": type_name},
    )


def open_slots(client: TestClient, type_name: str, day: date) -> list[dict]:
    """Return the Plateau's open slots for one appointment type on one day."""
    slots = get_slots(client, PLATEAU_ID, type_name, day, day).json()
    assert len(slots) >= 2, "these tests need at least two open slots"
    return slots


def open_slot_ids(client: TestClient, type_name: str, day: date) -> list[int]:
    ids = []
    for slot in get_slots(client, PLATEAU_ID, type_name, day, day).json():
        ids.append(slot["id"])
    return ids


def book_first_open_cleaning(client: TestClient, db_path: Path) -> dict:
    """Book the first open Plateau cleaning on FIRST_DAY and return it."""
    patient_id = patient_id_with_phone(db_path, MARIE_PHONE)
    slot = open_slots(client, "cleaning", FIRST_DAY)[0]
    response = book(client, patient_id, slot["id"], "cleaning")
    assert response.status_code == 201
    return response.json()


def start_of(slot: dict) -> datetime:
    return datetime.fromisoformat(slot["start_time"])


# --- POST /appointments ---


def test_book_open_slot_creates_appointment_and_takes_slot(client, db_path):
    patient_id = patient_id_with_phone(db_path, MARIE_PHONE)
    slot = open_slots(client, "cleaning", FIRST_DAY)[0]

    response = book(client, patient_id, slot["id"], "cleaning")

    assert response.status_code == 201
    appointment = response.json()
    assert appointment["patient_id"] == patient_id
    assert appointment["slot_id"] == slot["id"]
    assert appointment["type"] == "cleaning"
    assert appointment["status"] == "booked"
    assert appointment["start_time"] == slot["start_time"]
    assert appointment["location_name"] == "Plateau"
    assert slot["id"] not in open_slot_ids(client, "cleaning", FIRST_DAY)


def test_book_same_slot_twice_returns_409(client, db_path):
    first = book_first_open_cleaning(client, db_path)
    other_patient_id = patient_id_with_phone(db_path, HELENE_PHONE)

    response = book(client, other_patient_id, first["slot_id"], "cleaning")

    assert response.status_code == 409
    assert response.json() == {"detail": SLOT_TAKEN}


def test_database_rejects_double_booking_that_skips_the_check(client, db_path, monkeypatch):
    # Act out two requests that both passed check_slot_is_open at the same
    # moment: with the check switched off, the unique index must still
    # stop the second booking.
    first = book_first_open_cleaning(client, db_path)
    monkeypatch.setattr("backend.main.check_slot_is_open", lambda db, slot_id: None)
    other_patient_id = patient_id_with_phone(db_path, HELENE_PHONE)

    response = book(client, other_patient_id, first["slot_id"], "cleaning")

    assert response.status_code == 409
    assert response.json() == {"detail": SLOT_TAKEN}


def test_book_past_slot_returns_422(client, db_path):
    patient_id = patient_id_with_phone(db_path, MARIE_PHONE)
    slot = open_slots(client, "cleaning", FIRST_DAY)[0]
    set_now(start_of(slot) + timedelta(minutes=1))

    response = book(client, patient_id, slot["id"], "cleaning")

    assert response.status_code == 422
    assert response.json() == {"detail": "That slot is in the past."}
    # /slots hides slots that have started, so go back in time to check
    # that the refused booking didn't take the slot.
    set_now(TEST_NOW)
    assert slot["id"] in open_slot_ids(client, "cleaning", FIRST_DAY)


def test_book_beyond_four_weeks_returns_422(client, db_path):
    patient_id = patient_id_with_phone(db_path, MARIE_PHONE)
    slot = open_slots(client, "cleaning", FIRST_DAY)[0]
    # One minute earlier than "exactly four weeks before the slot".
    four_weeks_before = start_of(slot) - timedelta(days=BOOKING_WINDOW_DAYS)
    set_now(four_weeks_before - timedelta(minutes=1))

    response = book(client, patient_id, slot["id"], "cleaning")

    assert response.status_code == 422
    assert "at most 28 days ahead" in response.json()["detail"]


def test_book_exactly_four_weeks_ahead_is_allowed(client, db_path):
    patient_id = patient_id_with_phone(db_path, MARIE_PHONE)
    slot = open_slots(client, "cleaning", FIRST_DAY)[0]
    set_now(start_of(slot) - timedelta(days=BOOKING_WINDOW_DAYS))

    response = book(client, patient_id, slot["id"], "cleaning")

    assert response.status_code == 201


def test_book_checkup_in_hygienist_slot_returns_422(client, db_path):
    patient_id = patient_id_with_phone(db_path, MARIE_PHONE)
    cleaning_slot = open_slots(client, "cleaning", FIRST_DAY)[0]

    response = book(client, patient_id, cleaning_slot["id"], "checkup")

    assert response.status_code == 422


def test_book_unknown_patient_returns_404(client):
    slot = open_slots(client, "cleaning", FIRST_DAY)[0]
    response = book(client, 999_999, slot["id"], "cleaning")
    assert response.status_code == 404


def test_book_unknown_slot_returns_404(client, db_path):
    patient_id = patient_id_with_phone(db_path, MARIE_PHONE)
    response = book(client, patient_id, 999_999, "cleaning")
    assert response.status_code == 404


# --- PATCH /appointments/{id} ---


def test_reschedule_moves_appointment_and_frees_old_slot(client, db_path):
    appointment = book_first_open_cleaning(client, db_path)
    new_slot = open_slots(client, "cleaning", FIRST_DAY)[0]

    response = client.patch(
        f"/appointments/{appointment['id']}", json={"slot_id": new_slot["id"]}
    )

    assert response.status_code == 200
    assert response.json()["slot_id"] == new_slot["id"]
    assert response.json()["start_time"] == new_slot["start_time"]
    still_open = open_slot_ids(client, "cleaning", FIRST_DAY)
    assert appointment["slot_id"] in still_open
    assert new_slot["id"] not in still_open


def test_reschedule_to_taken_slot_returns_409(client, db_path):
    appointment = book_first_open_cleaning(client, db_path)
    other_slot = open_slots(client, "cleaning", FIRST_DAY)[0]
    other_patient_id = patient_id_with_phone(db_path, HELENE_PHONE)
    assert book(client, other_patient_id, other_slot["id"], "cleaning").status_code == 201

    response = client.patch(
        f"/appointments/{appointment['id']}", json={"slot_id": other_slot["id"]}
    )

    assert response.status_code == 409
    assert response.json() == {"detail": SLOT_TAKEN}


def test_reschedule_cancelled_appointment_returns_409(client, db_path):
    appointment = book_first_open_cleaning(client, db_path)
    client.delete(f"/appointments/{appointment['id']}", params={"reason": "Feeling unwell"})
    new_slot = open_slots(client, "cleaning", FIRST_DAY)[0]

    response = client.patch(
        f"/appointments/{appointment['id']}", json={"slot_id": new_slot["id"]}
    )

    assert response.status_code == 409


def test_reschedule_unknown_appointment_returns_404(client):
    slot = open_slots(client, "cleaning", FIRST_DAY)[0]
    response = client.patch("/appointments/999999", json={"slot_id": slot["id"]})
    assert response.status_code == 404


# --- DELETE /appointments/{id} ---


def test_cancel_frees_slot_and_stores_reason(client, db_path):
    appointment = book_first_open_cleaning(client, db_path)

    response = client.delete(
        f"/appointments/{appointment['id']}", params={"reason": "Scheduling conflict"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["cancel_reason"] == "Scheduling conflict"
    assert appointment["slot_id"] in open_slot_ids(client, "cleaning", FIRST_DAY)


def test_cancel_twice_returns_409(client, db_path):
    appointment = book_first_open_cleaning(client, db_path)
    url = f"/appointments/{appointment['id']}"
    client.delete(url, params={"reason": "Feeling unwell"})

    response = client.delete(url, params={"reason": "Feeling unwell"})

    assert response.status_code == 409


def test_cancel_without_reason_returns_422(client, db_path):
    appointment = book_first_open_cleaning(client, db_path)
    response = client.delete(f"/appointments/{appointment['id']}")
    assert response.status_code == 422


def test_cancel_unknown_appointment_returns_404(client):
    response = client.delete("/appointments/999999", params={"reason": "Travelling"})
    assert response.status_code == 404

import json
import sqlite3
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.tools import (
    MAX_SLOTS_RETURNED,
    NOT_VERIFIED,
    NOT_VERIFIED_PATIENTS_APPOINTMENT,
    CallState,
    book_appointment,
    cancel_appointment,
    check_availability,
    get_clinic_info,
    reschedule_appointment,
    run_tool,
    tool_definitions,
    verify_patient,
)
from backend.main import SLOT_TAKEN, app, connect, get_db, get_now
from backend.seed import fully_booked_day, seed

# A fixed Monday keeps these tests independent of the day they run.
FIRST_DAY = date(2026, 10, 26)
# Before the first slots open. /slots hides slots that have started, so a
# fixed "now" keeps these tests from failing once the real date passes them.
TEST_NOW = datetime(2026, 10, 26, 7, 0)

# Seeded patients (all fictional). Hélène is the only patient with this
# name, birthday and phone; Jean-François is a second, different patient.
HELENE = {"name": "Helene Tremblay", "date_of_birth": "1979-02-14", "phone": "514 555 0142"}
JEAN_FRANCOIS = {
    "name": "Jean-François Côté-Lavoie",
    "date_of_birth": "1990-11-23",
    "phone": "514-555-0104",
}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "clinic.db"
    seed(path, first_day=FIRST_DAY)
    return path


@pytest.fixture
def backend(db_path: Path) -> Iterator[TestClient]:
    """The real backend app on a freshly seeded temporary database."""

    def get_test_db() -> Iterator[sqlite3.Connection]:
        conn = connect(db_path)
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[get_db] = get_test_db
    app.dependency_overrides[get_now] = lambda: TEST_NOW
    # TestClient is a subclass of httpx.Client, so the tools can use it
    # exactly as they use the real client, without a server running.
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def unreachable_backend() -> httpx.Client:
    def refuse_connection(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    return httpx.Client(
        base_url="http://backend.test", transport=httpx.MockTransport(refuse_connection)
    )


# --- get_clinic_info ---


def test_clinic_info_verdun_is_closed_on_saturday(backend):
    result = get_clinic_info(backend, CallState(), "Verdun", "hours")
    assert result["ok"] is True
    assert result["details"]["saturday"] == "closed"
    assert result["details"]["monday"] == "08:00-17:00"
    assert result["open_elsewhere"]["saturday"] == {"Plateau": "09:00-13:00"}


def test_clinic_info_plateau_is_open_on_saturday(backend):
    result = get_clinic_info(backend, CallState(), "Plateau", "hours")
    assert result["details"]["saturday"] == "09:00-13:00"
    assert "saturday" not in result["open_elsewhere"]


def test_clinic_info_insurers_need_no_location(backend):
    result = get_clinic_info(backend, CallState(), None, "insurers")
    assert result["ok"] is True
    assert result["location"] is None
    assert "Sun Life" in result["details"]
    assert "open_elsewhere" not in result


def test_clinic_info_hours_without_location_is_an_error_result(backend):
    result = get_clinic_info(backend, CallState(), None, "hours")
    assert result["ok"] is False
    assert "Plateau, Verdun, Saint-Laurent" in result["error"]


def test_clinic_info_backend_down_is_an_error_result(unreachable_backend):
    result = get_clinic_info(unreachable_backend, CallState(), "Verdun", "hours")
    assert result == {
        "ok": False,
        "error": "The clinic's scheduling system can't be reached right now.",
    }


# --- check_availability ---


def test_availability_returns_open_slots_from_the_database(backend):
    result = check_availability(
        backend, CallState(), "Verdun", "cleaning", "2026-10-27", "2026-10-27"
    )

    assert result["ok"] is True
    assert result["open_slot_count"] > 0
    verdun_hygienists = {"Isabelle Côté", "Kevin Martin"}
    for slot in result["slots"]:
        assert slot["start_time"].startswith("2026-10-27")
        assert slot["provider"] in verdun_hygienists
        assert isinstance(slot["slot_id"], int)


def test_availability_lists_at_most_the_cap_and_says_so(backend):
    # A whole week of cleanings at Verdun is far more than the cap.
    result = check_availability(
        backend, CallState(), "Verdun", "cleaning", "2026-10-26", "2026-11-01"
    )

    assert len(result["slots"]) == MAX_SLOTS_RETURNED
    assert result["open_slot_count"] > MAX_SLOTS_RETURNED
    assert "narrower date range" in result["note"]


def test_availability_unknown_location_is_an_error_result(backend):
    result = check_availability(
        backend, CallState(), "Laval", "cleaning", "2026-10-26", "2026-10-30"
    )
    assert result == {"ok": False, "error": "There is no location named 'Laval'."}


def test_availability_bad_date_is_an_error_result(backend):
    result = check_availability(
        backend, CallState(), "Verdun", "cleaning", "next monday", "2026-10-30"
    )
    assert result["ok"] is False
    assert result["error"].startswith("Invalid request. from:")


def test_availability_backend_down_is_an_error_result(unreachable_backend):
    result = check_availability(
        unreachable_backend, CallState(), "Verdun", "cleaning", "2026-10-26", "2026-10-30"
    )
    assert result["ok"] is False


def test_availability_on_a_fully_booked_day_offers_alternatives(backend):
    full_day = fully_booked_day(FIRST_DAY).isoformat()  # Wednesday 2026-11-04

    result = check_availability(backend, CallState(), "Verdun", "cleaning", full_day, full_day)

    assert result["ok"] is True
    assert result["open_slot_count"] == 0
    alternatives = result["alternatives"]
    # The nearest open days at Verdun, on either side.
    for slot in alternatives["closest_earlier_day"]:
        assert slot["start_time"].startswith("2026-11-03")
    for slot in alternatives["closest_later_day"]:
        assert slot["start_time"].startswith("2026-11-05")
    # And the other locations, which are open that Wednesday.
    other_names = []
    for option in alternatives["other_locations"]:
        other_names.append(option["location"])
        assert len(option["slots"]) == 3
        assert option["slots"][0]["start_time"].startswith(full_day)
    assert other_names == ["Plateau", "Saint-Laurent"]


def test_availability_on_a_closed_saturday_offers_the_open_location(backend):
    result = check_availability(
        backend, CallState(), "Verdun", "cleaning", "2026-10-31", "2026-10-31"
    )

    alternatives = result["alternatives"]
    assert alternatives["closest_earlier_day"][0]["start_time"].startswith("2026-10-30")
    assert alternatives["closest_later_day"][0]["start_time"].startswith("2026-11-02")
    # Saint-Laurent is closed on Saturdays too, so only Plateau is offered.
    assert len(alternatives["other_locations"]) == 1
    assert alternatives["other_locations"][0]["location"] == "Plateau"


def test_availability_leaves_out_a_side_with_nothing_open(backend):
    # The day before the first seeded day: nothing earlier exists at all.
    result = check_availability(
        backend, CallState(), "Verdun", "cleaning", "2026-10-25", "2026-10-25"
    )

    alternatives = result["alternatives"]
    assert "closest_earlier_day" not in alternatives
    assert alternatives["closest_later_day"][0]["start_time"].startswith("2026-10-26")


def test_availability_with_open_slots_has_no_alternatives(backend):
    result = check_availability(
        backend, CallState(), "Verdun", "cleaning", "2026-11-03", "2026-11-03"
    )
    assert result["open_slot_count"] > 0
    assert "alternatives" not in result


# --- verify_patient ---


def test_verify_patient_match_marks_the_patient_verified(backend):
    call_state = CallState()

    result = verify_patient(backend, call_state, **HELENE)

    assert result["ok"] is True
    assert call_state.verified_patient_ids == {result["patient_id"]}


def test_verify_patient_wrong_detail_verifies_no_one(backend):
    call_state = CallState()
    wrong_phone = dict(HELENE, phone="514-555-0199")

    result = verify_patient(backend, call_state, **wrong_phone)

    assert result["ok"] is False
    assert "No patient matches" in result["error"]
    assert call_state.verified_patient_ids == set()


def test_verify_patient_backend_down_is_an_error_result(unreachable_backend):
    call_state = CallState()
    result = verify_patient(unreachable_backend, call_state, **HELENE)
    assert result["ok"] is False
    assert call_state.verified_patient_ids == set()


# --- book_appointment ---


def count_booked(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE status = 'booked'"
        ).fetchone()[0]
    finally:
        conn.close()


def first_open_verdun_cleaning(backend: TestClient) -> dict:
    result = check_availability(
        backend, CallState(), "Verdun", "cleaning", "2026-10-27", "2026-10-27"
    )
    return result["slots"][0]


def test_book_after_verifying_creates_the_appointment(backend, db_path):
    call_state = CallState()
    patient_id = verify_patient(backend, call_state, **HELENE)["patient_id"]
    slot = first_open_verdun_cleaning(backend)

    result = book_appointment(backend, call_state, patient_id, slot["slot_id"], "cleaning")

    assert result["ok"] is True
    assert result["location"] == "Verdun"
    assert result["start_time"] == slot["start_time"]
    assert result["provider"] == slot["provider"]
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT patient_id, slot_id, status FROM appointments WHERE id = ?",
        (result["appointment_id"],),
    ).fetchone()
    conn.close()
    assert row == (patient_id, slot["slot_id"], "booked")


def test_book_without_verifying_is_refused_and_changes_nothing(backend, db_path):
    patient_id = verify_patient(backend, CallState(), **HELENE)["patient_id"]
    slot = first_open_verdun_cleaning(backend)
    booked_before = count_booked(db_path)

    # A real patient id, but verified on a different call.
    result = book_appointment(backend, CallState(), patient_id, slot["slot_id"], "cleaning")

    assert result == {"ok": False, "error": NOT_VERIFIED}
    assert count_booked(db_path) == booked_before


def test_book_for_a_different_patient_than_the_verified_one_is_refused(backend, db_path):
    call_state = CallState()
    verify_patient(backend, call_state, **HELENE)
    other_patient_id = verify_patient(backend, CallState(), **JEAN_FRANCOIS)["patient_id"]
    slot = first_open_verdun_cleaning(backend)
    booked_before = count_booked(db_path)

    result = book_appointment(backend, call_state, other_patient_id, slot["slot_id"], "cleaning")

    assert result == {"ok": False, "error": NOT_VERIFIED}
    assert count_booked(db_path) == booked_before


def test_book_a_taken_slot_is_an_error_result(backend):
    call_state = CallState()
    patient_id = verify_patient(backend, call_state, **HELENE)["patient_id"]
    slot = first_open_verdun_cleaning(backend)
    book_appointment(backend, call_state, patient_id, slot["slot_id"], "cleaning")

    result = book_appointment(backend, call_state, patient_id, slot["slot_id"], "cleaning")

    assert result == {"ok": False, "error": SLOT_TAKEN}


# --- reschedule_appointment and cancel_appointment ---


def book_helene_on_an_earlier_call(backend: TestClient) -> dict:
    """Book Hélène's first open Verdun cleaning, as if on an earlier call."""
    # Its own CallState, so the tests below start a new call in which
    # nobody is verified yet, like a patient phoning back.
    earlier_call = CallState()
    patient_id = verify_patient(backend, earlier_call, **HELENE)["patient_id"]
    slot = first_open_verdun_cleaning(backend)
    return book_appointment(backend, earlier_call, patient_id, slot["slot_id"], "cleaning")


def appointment_row(db_path: Path, appointment_id: int) -> tuple:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT slot_id, status, cancel_reason FROM appointments WHERE id = ?",
            (appointment_id,),
        ).fetchone()
    finally:
        conn.close()


def test_verify_patient_lists_upcoming_appointments(backend):
    appointment = book_helene_on_an_earlier_call(backend)

    result = verify_patient(backend, CallState(), **HELENE)

    listed_ids = []
    for upcoming in result["upcoming_appointments"]:
        listed_ids.append(upcoming["appointment_id"])
    assert appointment["appointment_id"] in listed_ids


def test_reschedule_after_verifying_moves_the_appointment(backend, db_path):
    appointment = book_helene_on_an_earlier_call(backend)
    call_state = CallState()
    verify_patient(backend, call_state, **HELENE)
    new_slot = first_open_verdun_cleaning(backend)

    result = reschedule_appointment(
        backend, call_state, appointment["appointment_id"], new_slot["slot_id"]
    )

    assert result["ok"] is True
    assert result["start_time"] == new_slot["start_time"]
    assert appointment_row(db_path, appointment["appointment_id"]) == (
        new_slot["slot_id"], "booked", None,
    )


def test_cancel_after_verifying_cancels_and_stores_the_reason(backend, db_path):
    appointment = book_helene_on_an_earlier_call(backend)
    call_state = CallState()
    verify_patient(backend, call_state, **HELENE)
    slot_id = appointment_row(db_path, appointment["appointment_id"])[0]

    result = cancel_appointment(
        backend, call_state, appointment["appointment_id"], "Scheduling conflict"
    )

    assert result["ok"] is True
    assert result["status"] == "cancelled"
    assert appointment_row(db_path, appointment["appointment_id"]) == (
        slot_id, "cancelled", "Scheduling conflict",
    )


def test_reschedule_and_cancel_without_verifying_change_nothing(backend, db_path):
    appointment = book_helene_on_an_earlier_call(backend)
    row_before = appointment_row(db_path, appointment["appointment_id"])
    new_slot = first_open_verdun_cleaning(backend)
    call_state = CallState()

    moved = reschedule_appointment(
        backend, call_state, appointment["appointment_id"], new_slot["slot_id"]
    )
    cancelled = cancel_appointment(backend, call_state, appointment["appointment_id"], "Travel")

    assert moved == {"ok": False, "error": NOT_VERIFIED}
    assert cancelled == {"ok": False, "error": NOT_VERIFIED}
    assert appointment_row(db_path, appointment["appointment_id"]) == row_before


def test_reschedule_and_cancel_after_failed_verification_change_nothing(backend, db_path):
    appointment = book_helene_on_an_earlier_call(backend)
    row_before = appointment_row(db_path, appointment["appointment_id"])
    new_slot = first_open_verdun_cleaning(backend)
    call_state = CallState()
    # Right name and birthday, wrong phone: the caller doesn't get through.
    assert verify_patient(backend, call_state, **dict(HELENE, phone="514-555-0199"))["ok"] is False

    moved = reschedule_appointment(
        backend, call_state, appointment["appointment_id"], new_slot["slot_id"]
    )
    cancelled = cancel_appointment(backend, call_state, appointment["appointment_id"], "Travel")

    assert moved == {"ok": False, "error": NOT_VERIFIED}
    assert cancelled == {"ok": False, "error": NOT_VERIFIED}
    assert appointment_row(db_path, appointment["appointment_id"]) == row_before


def test_reschedule_and_cancel_someone_elses_appointment_change_nothing(backend, db_path):
    appointment = book_helene_on_an_earlier_call(backend)
    row_before = appointment_row(db_path, appointment["appointment_id"])
    new_slot = first_open_verdun_cleaning(backend)
    call_state = CallState()
    verify_patient(backend, call_state, **JEAN_FRANCOIS)  # verified, but not Hélène

    moved = reschedule_appointment(
        backend, call_state, appointment["appointment_id"], new_slot["slot_id"]
    )
    cancelled = cancel_appointment(backend, call_state, appointment["appointment_id"], "Travel")

    assert moved == {"ok": False, "error": NOT_VERIFIED_PATIENTS_APPOINTMENT}
    assert cancelled == {"ok": False, "error": NOT_VERIFIED_PATIENTS_APPOINTMENT}
    assert appointment_row(db_path, appointment["appointment_id"]) == row_before


def test_cancel_unknown_appointment_is_an_error_result(backend):
    call_state = CallState()
    verify_patient(backend, call_state, **HELENE)

    result = cancel_appointment(backend, call_state, 999_999, "Travel")

    assert result == {"ok": False, "error": "No appointment with id 999999."}


def test_cancel_twice_is_an_error_result(backend):
    appointment = book_helene_on_an_earlier_call(backend)
    call_state = CallState()
    verify_patient(backend, call_state, **HELENE)
    cancel_appointment(backend, call_state, appointment["appointment_id"], "Travel")

    result = cancel_appointment(backend, call_state, appointment["appointment_id"], "Travel")

    assert result == {"ok": False, "error": "That appointment is already cancelled."}


# --- run_tool and the definitions ---


def test_run_tool_refuses_a_booking_the_model_asks_for_without_verifying(backend, db_path):
    # What a model that skipped verification would send: a plausible
    # patient id and a real open slot.
    slot = first_open_verdun_cleaning(backend)
    arguments = json.dumps(
        {"patient_id": 1, "slot_id": slot["slot_id"], "appointment_type": "cleaning"}
    )
    booked_before = count_booked(db_path)

    result = run_tool(backend, CallState(), "book_appointment", arguments)

    assert result == {"ok": False, "error": NOT_VERIFIED}
    assert count_booked(db_path) == booked_before


def test_run_tool_refuses_a_cancel_the_model_asks_for_without_verifying(backend, db_path):
    appointment = book_helene_on_an_earlier_call(backend)
    row_before = appointment_row(db_path, appointment["appointment_id"])
    arguments = json.dumps({"appointment_id": appointment["appointment_id"], "reason": "Travel"})

    result = run_tool(backend, CallState(), "cancel_appointment", arguments)

    assert result == {"ok": False, "error": NOT_VERIFIED}
    assert appointment_row(db_path, appointment["appointment_id"]) == row_before


def test_run_tool_reschedules_for_a_verified_patient(backend, db_path):
    appointment = book_helene_on_an_earlier_call(backend)
    call_state = CallState()
    verify_patient(backend, call_state, **HELENE)
    new_slot = first_open_verdun_cleaning(backend)
    arguments = json.dumps(
        {"appointment_id": appointment["appointment_id"], "new_slot_id": new_slot["slot_id"]}
    )

    result = run_tool(backend, call_state, "reschedule_appointment", arguments)

    assert result["ok"] is True
    assert appointment_row(db_path, appointment["appointment_id"])[0] == new_slot["slot_id"]


def test_run_tool_parses_arguments_and_calls_the_tool(backend):
    arguments = json.dumps({"location": "Verdun", "topic": "hours"})
    result = run_tool(backend, CallState(), "get_clinic_info", arguments)
    assert result["details"]["saturday"] == "closed"


def test_run_tool_unknown_tool_is_an_error_result(backend):
    result = run_tool(backend, CallState(), "book_spa_day", "{}")
    assert result == {"ok": False, "error": "There is no tool named 'book_spa_day'."}


def test_run_tool_invalid_json_is_an_error_result(backend):
    result = run_tool(backend, CallState(), "get_clinic_info", "{not json")
    assert result == {"ok": False, "error": "The tool arguments were not valid JSON."}


def test_run_tool_wrong_arguments_are_an_error_result(backend):
    result = run_tool(backend, CallState(), "get_clinic_info", json.dumps({"place": "Verdun"}))
    assert result["ok"] is False
    assert result["error"].startswith("The tool failed unexpectedly: TypeError")


def test_tool_definitions_offer_only_the_configured_locations():
    definitions = tool_definitions(["Plateau", "Verdun"])
    names = []
    for definition in definitions:
        names.append(definition["name"])
    assert names == [
        "get_clinic_info", "check_availability", "verify_patient", "book_appointment",
        "reschedule_appointment", "cancel_appointment",
    ]

    availability = definitions[1]["parameters"]["properties"]
    assert availability["location"]["enum"] == ["Plateau", "Verdun"]

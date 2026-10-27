import sqlite3
from datetime import date, datetime
from pathlib import Path

import pytest

from backend.seed import seed
from evals.availability import (
    check_offered_times,
    find_open_slots,
    resolve_day,
    resolve_request,
)

TUESDAY = date(2026, 10, 27)
# A fixed Monday for the seeded database, as in test_seed.py.
FIRST_DAY = date(2026, 10, 26)
START_OF_FIRST_DAY = datetime(2026, 10, 26, 0, 0)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "clinic.db"
    seed(path, first_day=FIRST_DAY)
    return path


# --- resolve_day and resolve_request ---


def test_resolve_day_tomorrow():
    assert resolve_day("tomorrow", TUESDAY) == date(2026, 10, 28)


def test_resolve_day_next_weekday_is_in_next_week():
    # From Tuesday the 27th, next week is Monday Nov 2 to Sunday Nov 8.
    assert resolve_day("next monday", TUESDAY) == date(2026, 11, 2)
    assert resolve_day("next thursday", TUESDAY) == date(2026, 11, 5)
    assert resolve_day("next sunday", TUESDAY) == date(2026, 11, 8)


def test_resolve_day_on_a_monday_skips_the_current_week():
    assert resolve_day("next monday", date(2026, 10, 26)) == date(2026, 11, 2)


def test_resolve_day_rejects_an_unknown_word():
    with pytest.raises(ValueError, match="next fortnight"):
        resolve_day("next fortnight", TUESDAY)


def test_resolve_request_keeps_the_rest_of_the_request():
    availability = {
        "location": "Verdun",
        "appointment_type": "checkup",
        "first_day": "next thursday",
        "last_day": "next thursday",
        "earliest_start": "12:00",
    }
    request = resolve_request(availability, TUESDAY)
    assert request["first_day"] == "2026-11-05"
    assert request["last_day"] == "2026-11-05"
    assert request["earliest_start"] == "12:00"
    assert availability["first_day"] == "next thursday"  # the case is not changed


# --- find_open_slots ---


def test_find_open_slots_matches_provider_role_and_opening_hours(db_path):
    cleanings = find_open_slots(db_path, "cleaning", START_OF_FIRST_DAY)
    checkups = find_open_slots(db_path, "checkup", START_OF_FIRST_DAY)

    # Plateau is open Saturday mornings; Verdun is closed Saturdays.
    assert ("Plateau", "2026-10-31T09:00") in cleanings
    for location, start_time in cleanings:
        assert not (location == "Verdun" and start_time.startswith("2026-10-31"))
    # Cleanings are 60 minutes on the hour; checkups 30 minutes.
    assert ("Saint-Laurent", "2026-11-04T10:30") not in cleanings
    assert ("Saint-Laurent", "2026-11-04T10:30") in checkups


def book_cleaning(db_path: Path, location: str, start_time: str, status: str) -> None:
    conn = sqlite3.connect(db_path)
    # Dentists have slots at the same times, so pick the hygienist's.
    slot_id = conn.execute(
        "SELECT slots.id FROM slots"
        " JOIN locations ON locations.id = slots.location_id"
        " JOIN providers ON providers.id = slots.provider_id"
        " WHERE locations.name = ? AND slots.start_time = ? AND providers.role = 'hygienist'",
        (location, start_time),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO appointments"
        " (patient_id, slot_id, appointment_type_id, status, created_at)"
        " VALUES (1, ?, (SELECT id FROM appointment_types WHERE name = 'cleaning'), ?, ?)",
        (slot_id, status, "2026-10-25T10:00:00"),
    )
    conn.commit()
    conn.close()


def test_find_open_slots_leaves_out_booked_slots(db_path):
    # Saint-Laurent has one hygienist, so this time has one cleaning slot.
    # (Where two providers share a time, it stays open while either is free.)
    before = find_open_slots(db_path, "cleaning", START_OF_FIRST_DAY)
    book_cleaning(db_path, "Saint-Laurent", "2026-11-03T09:00", "booked")

    after = find_open_slots(db_path, "cleaning", START_OF_FIRST_DAY)

    assert ("Saint-Laurent", "2026-11-03T09:00") in before
    assert ("Saint-Laurent", "2026-11-03T09:00") not in after


def test_find_open_slots_keeps_a_slot_whose_appointment_was_cancelled(db_path):
    book_cleaning(db_path, "Saint-Laurent", "2026-11-03T10:00", "cancelled")
    open_slots = find_open_slots(db_path, "cleaning", START_OF_FIRST_DAY)
    assert ("Saint-Laurent", "2026-11-03T10:00") in open_slots


def test_find_open_slots_leaves_out_slots_that_have_started(db_path):
    noon = datetime(2026, 10, 26, 12, 0)

    open_slots = find_open_slots(db_path, "cleaning", noon)

    assert ("Plateau", "2026-10-26T13:00") in open_slots
    for _location, start_time in open_slots:
        assert start_time >= "2026-10-26T12:00"


# --- check_offered_times ---


THURSDAY_AFTERNOON = {
    "location": "Verdun",
    "appointment_type": "checkup",
    "first_day": "2026-11-05",
    "last_day": "2026-11-05",
    "earliest_start": "12:00",
}
OPEN_SLOTS = {
    ("Verdun", "2026-11-05T09:00"),
    ("Verdun", "2026-11-05T13:00"),
    ("Verdun", "2026-11-05T14:30"),
    ("Plateau", "2026-11-05T13:00"),
}


def offer(day: str, start_time: str, location: str = "Verdun") -> dict:
    return {"date": day, "start_time": start_time, "location": location}


def test_offered_times_pass_when_real_and_inside_the_request():
    offered = [offer("2026-11-05", "13:00"), offer("2026-11-05", "14:30")]
    result = check_offered_times(offered, False, THURSDAY_AFTERNOON, OPEN_SLOTS)
    assert result["passed"]


def test_an_invented_time_fails():
    offered = [offer("2026-11-05", "15:00")]
    result = check_offered_times(offered, False, THURSDAY_AFTERNOON, OPEN_SLOTS)
    assert not result["passed"]
    assert "not an open checkup slot" in result["reason"]


def test_a_real_time_outside_the_request_fails_while_matching_times_are_open():
    offered = [offer("2026-11-05", "09:00")]  # real, but morning
    result = check_offered_times(offered, False, THURSDAY_AFTERNOON, OPEN_SLOTS)
    assert not result["passed"]
    assert "outside the request" in result["reason"]


def test_saying_unavailable_fails_when_matching_times_are_open():
    result = check_offered_times([], True, THURSDAY_AFTERNOON, OPEN_SLOTS)
    assert not result["passed"]
    assert "said the request was unavailable" in result["reason"]


def test_offering_nothing_fails_when_matching_times_are_open():
    result = check_offered_times([], False, THURSDAY_AFTERNOON, OPEN_SLOTS)
    assert not result["passed"]
    assert "offered no times" in result["reason"]


def test_when_nothing_matches_real_alternatives_pass_if_the_reply_says_so():
    saturday_at_verdun = dict(THURSDAY_AFTERNOON, first_day="2026-11-07", last_day="2026-11-07")
    alternatives = [offer("2026-11-05", "13:00", "Plateau")]

    honest = check_offered_times(alternatives, True, saturday_at_verdun, OPEN_SLOTS)
    pretending = check_offered_times(alternatives, False, saturday_at_verdun, OPEN_SLOTS)

    assert honest["passed"]
    assert not pretending["passed"]

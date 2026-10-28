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
    check_availability,
    get_clinic_info,
    run_tool,
    tool_definitions,
)
from backend.main import app, connect, get_db, get_now
from backend.seed import seed

# A fixed Monday keeps these tests independent of the day they run.
FIRST_DAY = date(2026, 10, 26)
# Before the first slots open. /slots hides slots that have started, so a
# fixed "now" keeps these tests from failing once the real date passes them.
TEST_NOW = datetime(2026, 10, 26, 7, 0)


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[TestClient]:
    """The real backend app on a freshly seeded temporary database."""
    db_path = tmp_path / "clinic.db"
    seed(db_path, first_day=FIRST_DAY)

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
    result = get_clinic_info(backend, "Verdun", "hours")
    assert result["ok"] is True
    assert result["details"]["saturday"] == "closed"
    assert result["details"]["monday"] == "08:00-17:00"
    assert result["open_elsewhere"]["saturday"] == {"Plateau": "09:00-13:00"}


def test_clinic_info_plateau_is_open_on_saturday(backend):
    result = get_clinic_info(backend, "Plateau", "hours")
    assert result["details"]["saturday"] == "09:00-13:00"
    assert "saturday" not in result["open_elsewhere"]


def test_clinic_info_insurers_need_no_location(backend):
    result = get_clinic_info(backend, None, "insurers")
    assert result["ok"] is True
    assert result["location"] is None
    assert "Sun Life" in result["details"]
    assert "open_elsewhere" not in result


def test_clinic_info_hours_without_location_is_an_error_result(backend):
    result = get_clinic_info(backend, None, "hours")
    assert result["ok"] is False
    assert "Plateau, Verdun, Saint-Laurent" in result["error"]


def test_clinic_info_backend_down_is_an_error_result(unreachable_backend):
    result = get_clinic_info(unreachable_backend, "Verdun", "hours")
    assert result == {
        "ok": False,
        "error": "The clinic's scheduling system can't be reached right now.",
    }


# --- check_availability ---


def test_availability_returns_open_slots_from_the_database(backend):
    result = check_availability(backend, "Verdun", "cleaning", "2026-10-27", "2026-10-27")

    assert result["ok"] is True
    assert result["open_slot_count"] > 0
    verdun_hygienists = {"Isabelle Côté", "Kevin Martin"}
    for slot in result["slots"]:
        assert slot["start_time"].startswith("2026-10-27")
        assert slot["provider"] in verdun_hygienists
        assert isinstance(slot["slot_id"], int)


def test_availability_lists_at_most_the_cap_and_says_so(backend):
    # A whole week of cleanings at Verdun is far more than the cap.
    result = check_availability(backend, "Verdun", "cleaning", "2026-10-26", "2026-11-01")

    assert len(result["slots"]) == MAX_SLOTS_RETURNED
    assert result["open_slot_count"] > MAX_SLOTS_RETURNED
    assert "open_slots_by_day counts them all" in result["note"]


def test_availability_names_the_weekday_of_each_slot(backend):
    result = check_availability(backend, "Verdun", "cleaning", "2026-10-27", "2026-10-27")
    for slot in result["slots"]:
        assert slot["day"] == "Tuesday"


def test_availability_counts_every_day_searched_and_marks_closed_days(backend):
    result = check_availability(backend, "Verdun", "cleaning", "2026-10-26", "2026-11-01")
    by_day = result["open_slots_by_day"]

    # The listed slots stop early in the week, but every day is counted,
    # so a Thursday with open slots can't be mistaken for a closed one.
    for slot in result["slots"]:
        assert not slot["start_time"].startswith("2026-10-29")
    assert by_day["Thursday 2026-10-29"] > 0
    assert by_day["Saturday 2026-10-31"] == "closed"
    assert by_day["Sunday 2026-11-01"] == "closed"

    total = 0
    for count in by_day.values():
        if count != "closed":
            total += count
    assert total == result["open_slot_count"]


def test_availability_says_where_else_is_open_on_closed_days(backend):
    week = check_availability(backend, "Verdun", "cleaning", "2026-10-26", "2026-11-01")
    weekdays = check_availability(backend, "Verdun", "cleaning", "2026-10-26", "2026-10-30")

    assert week["open_elsewhere"] == {
        "Saturday 2026-10-31": {"Plateau": "09:00-13:00"},
        "Sunday 2026-11-01": {},
    }
    assert "open_elsewhere" not in weekdays


def test_availability_between_start_times(backend):
    result = check_availability(
        backend, "Verdun", "checkup", "2026-10-29", "2026-10-29", "12:00", "17:00"
    )

    assert result["open_slot_count"] > 0
    for slot in result["slots"]:
        assert slot["start_time"][11:16] >= "12:00"
    # The day's count follows the same limits.
    assert result["open_slots_by_day"] == {"Thursday 2026-10-29": result["open_slot_count"]}


def test_availability_unknown_location_is_an_error_result(backend):
    result = check_availability(backend, "Laval", "cleaning", "2026-10-26", "2026-10-30")
    assert result == {"ok": False, "error": "There is no location named 'Laval'."}


def test_availability_bad_date_is_an_error_result(backend):
    result = check_availability(backend, "Verdun", "cleaning", "next monday", "2026-10-30")
    assert result["ok"] is False
    assert result["error"].startswith("Invalid request. from:")


def test_availability_backend_down_is_an_error_result(unreachable_backend):
    result = check_availability(
        unreachable_backend, "Verdun", "cleaning", "2026-10-26", "2026-10-30"
    )
    assert result["ok"] is False


# --- run_tool and the definitions ---


def test_run_tool_parses_arguments_and_calls_the_tool(backend):
    arguments = json.dumps({"location": "Verdun", "topic": "hours"})
    result = run_tool(backend, "get_clinic_info", arguments)
    assert result["details"]["saturday"] == "closed"


def test_run_tool_unknown_tool_is_an_error_result(backend):
    result = run_tool(backend, "book_spa_day", "{}")
    assert result == {"ok": False, "error": "There is no tool named 'book_spa_day'."}


def test_run_tool_invalid_json_is_an_error_result(backend):
    result = run_tool(backend, "get_clinic_info", "{not json")
    assert result == {"ok": False, "error": "The tool arguments were not valid JSON."}


def test_run_tool_wrong_arguments_are_an_error_result(backend):
    result = run_tool(backend, "get_clinic_info", json.dumps({"place": "Verdun"}))
    assert result["ok"] is False
    assert result["error"].startswith("The tool failed unexpectedly: TypeError")


def test_tool_definitions_offer_only_the_configured_locations():
    definitions = tool_definitions(["Plateau", "Verdun"])
    names = []
    for definition in definitions:
        names.append(definition["name"])
    assert names == ["get_clinic_info", "check_availability"]

    availability = definitions[1]["parameters"]["properties"]
    assert availability["location"]["enum"] == ["Plateau", "Verdun"]


def test_run_tool_availability_with_null_start_times(backend):
    # Strict mode sends every parameter, so "no limit" arrives as null.
    arguments = json.dumps(
        {
            "location": "Verdun",
            "appointment_type": "cleaning",
            "first_day": "2026-10-27",
            "last_day": "2026-10-27",
            "earliest_start": None,
            "latest_start": None,
        }
    )
    result = run_tool(backend, "check_availability", arguments)
    assert result["ok"] is True
    assert result["open_slot_count"] > 0


def test_tool_definitions_require_every_property_for_strict_mode():
    # The API rejects a strict tool whose "required" leaves a property out.
    for definition in tool_definitions(["Plateau"]):
        parameters = definition["parameters"]
        assert parameters["required"] == list(parameters["properties"])

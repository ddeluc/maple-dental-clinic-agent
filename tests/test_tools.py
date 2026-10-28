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
    assert "narrower date range" in result["note"]


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

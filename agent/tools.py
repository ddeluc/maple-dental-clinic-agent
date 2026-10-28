"""Tools the model can call, shared by the text agent and the voice agent.

Each tool calls the fake practice management system (backend/) over HTTP.
Every tool returns a dict the model can read, and failures come back as
{"ok": False, "error": "..."} instead of exceptions. The model can then
explain the problem to the caller, and the conversation keeps going.
"""

import json
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import httpx

BACKEND_URL = "http://localhost:8000"
BACKEND_TIMEOUT_SECONDS = 5.0

CLINIC_INFO_TOPICS = ["hours", "address", "parking", "insurers"]
# Must match the appointment types the backend knows (backend/seed.py).
APPOINTMENT_TYPES = ["cleaning", "checkup"]

# A week of cleanings at one location is about 80 slots. Sending them all
# would cost tokens and slow the reply, and the caller only hears two or
# three, so the model gets the earliest ones and is told how many exist.
MAX_SLOTS_RETURNED = 20

# How the backend writes the hours of a day a location is closed.
CLOSED = "closed"

ToolResult = dict[str, Any]


def describe_day(day: date) -> str:
    """A day as the model sees it everywhere: "Monday 2026-11-02"."""
    # The weekday is always spelled out next to the date, in the calendar
    # in the instructions and in tool results alike. Left to work out a
    # weekday from a date, the model gets it wrong.
    return f"{day.strftime('%A')} {day.isoformat()}"


def tool_error(message: str) -> ToolResult:
    return {"ok": False, "error": message}


def make_backend_client() -> httpx.Client:
    # One client for the whole conversation reuses the same connection
    # instead of opening a new one for every tool call.
    return httpx.Client(base_url=BACKEND_URL, timeout=BACKEND_TIMEOUT_SECONDS)


def error_detail(response: httpx.Response) -> str:
    """Pull FastAPI's error message out of a failed response."""
    try:
        detail = response.json()["detail"]
    except (ValueError, KeyError, TypeError):  # not JSON, or no "detail" key
        return f"The clinic system returned an error (HTTP {response.status_code})."

    # HTTPException gives a plain message. FastAPI's own validation errors
    # (for example a badly written date) give a list of problems instead.
    if isinstance(detail, str):
        return detail
    messages = []
    for problem in detail:
        field = problem["loc"][-1]
        messages.append(f"{field}: {problem['msg']}")
    return "Invalid request. " + "; ".join(messages)


def get_from_backend(
    backend: httpx.Client, path: str, params: dict[str, Any] | None = None
) -> ToolResult:
    """GET a backend path; return {"ok": True, "data": ...} or an error result."""
    try:
        response = backend.get(path, params=params)
    except httpx.HTTPError:
        # Connection refused, timeout, DNS failure: the backend isn't there.
        return tool_error("The clinic's scheduling system can't be reached right now.")

    # httpx doesn't raise on 4xx/5xx unless asked to, so check the status.
    if response.is_error:
        return tool_error(error_detail(response))
    return {"ok": True, "data": response.json()}


# --- The tools ---


def get_clinic_info(backend: httpx.Client, location: str | None, topic: str) -> ToolResult:
    params = {"topic": topic}
    # Leaving location out of the URL is how the backend knows the
    # question is clinic-wide (insurers).
    if location is not None:
        params["location"] = location

    result = get_from_backend(backend, "/clinic-info", params)
    if not result["ok"]:
        return result
    info = result["data"]
    tool_result = {
        "ok": True,
        "topic": info["topic"],
        "location": info["location"],
        "details": info["details"],
    }
    # Only hours have it, and leaving it out otherwise keeps the result short.
    if info["open_elsewhere"] is not None:
        tool_result["open_elsewhere"] = info["open_elsewhere"]
    return tool_result


def find_location_id(backend: httpx.Client, location: str) -> ToolResult:
    """Turn a location name into the id /slots expects."""
    result = get_from_backend(backend, "/locations")
    if not result["ok"]:
        return result
    for known_location in result["data"]:
        if known_location["name"] == location:
            return {"ok": True, "location_id": known_location["id"]}
    return tool_error(f"There is no location named '{location}'.")


def count_open_slots_by_day(
    first_day: date,
    last_day: date,
    all_slots: list[dict[str, Any]],
    hours_info: dict[str, Any],
) -> dict[str, int | str]:
    """How many slots are open on each day searched, or "closed"."""
    # Every day searched is listed, even past the slots shown. Without this,
    # a week's search shows only Monday's slots, and the model concluded
    # the location was closed on the Thursday the caller asked for.
    counts_by_date: dict[str, int] = {}
    for slot in all_slots:
        slot_date = slot["start_time"][:10]
        counts_by_date[slot_date] = counts_by_date.get(slot_date, 0) + 1

    days: dict[str, int | str] = {}
    day = first_day
    while day <= last_day:
        weekday = day.strftime("%A").lower()  # the hours use "monday"
        if hours_info["details"][weekday] == CLOSED:
            days[describe_day(day)] = CLOSED
        else:
            days[describe_day(day)] = counts_by_date.get(day.isoformat(), 0)
        day += timedelta(days=1)
    return days


def open_elsewhere_on_closed_days(
    first_day: date, last_day: date, hours_info: dict[str, Any]
) -> dict[str, dict[str, str]]:
    """For each closed day searched, the other locations open that day."""
    open_elsewhere = {}
    day = first_day
    while day <= last_day:
        weekday = day.strftime("%A").lower()
        if hours_info["details"][weekday] == CLOSED:
            open_elsewhere[describe_day(day)] = hours_info["open_elsewhere"][weekday]
        day += timedelta(days=1)
    return open_elsewhere


def check_availability(
    backend: httpx.Client,
    location: str,
    appointment_type: str,
    first_day: str,
    last_day: str,
    earliest_start: str | None = None,
    latest_start: str | None = None,
) -> ToolResult:
    location_result = find_location_id(backend, location)
    if not location_result["ok"]:
        return location_result

    params = {
        "location": location_result["location_id"],
        "type": appointment_type,
        "from": first_day,
        "to": last_day,
    }
    # Left out of the URL when null, which the backend reads as "no limit".
    if earliest_start is not None:
        params["earliest_start"] = earliest_start
    if latest_start is not None:
        params["latest_start"] = latest_start
    slots_result = get_from_backend(backend, "/slots", params)
    if not slots_result["ok"]:
        return slots_result

    # The hours say which days are closed, so the day counts can tell
    # "closed" apart from "open, but nothing free".
    hours_result = get_from_backend(
        backend, "/clinic-info", {"topic": "hours", "location": location}
    )
    if not hours_result["ok"]:
        return hours_result
    hours_info = hours_result["data"]

    all_slots = slots_result["data"]
    # Only what the model needs to offer and later book a slot. Fewer
    # fields means fewer tokens and a shorter debug line.
    slots = []
    for slot in all_slots[:MAX_SLOTS_RETURNED]:
        start_day = date.fromisoformat(slot["start_time"][:10])
        slots.append(
            {
                "slot_id": slot["id"],
                "day": start_day.strftime("%A"),
                "start_time": slot["start_time"],
                "provider": slot["provider_name"],
            }
        )

    # The backend has already rejected most bad dates with a clear message,
    # but it also accepts a few forms, such as timestamps, that
    # date.fromisoformat doesn't.
    try:
        first = date.fromisoformat(first_day)
        last = date.fromisoformat(last_day)
    except ValueError:
        return tool_error("Dates must look like YYYY-MM-DD, such as 2026-11-02.")
    result = {
        "ok": True,
        "location": location,
        "appointment_type": appointment_type,
        "open_slot_count": len(all_slots),
        "open_slots_by_day": count_open_slots_by_day(first, last, all_slots, hours_info),
        "slots": slots,
    }
    open_elsewhere = open_elsewhere_on_closed_days(first, last, hours_info)
    if open_elsewhere:
        result["open_elsewhere"] = open_elsewhere
    if len(all_slots) > MAX_SLOTS_RETURNED:
        result["note"] = (
            f"Only the earliest {MAX_SLOTS_RETURNED} are listed; open_slots_by_day"
            " counts them all. Search one day, or a narrower range of start"
            " times, to see later ones."
        )
    return result


# --- What the model sees, and how its requests are dispatched ---

TOOL_FUNCTIONS: dict[str, Callable[..., ToolResult]] = {
    "get_clinic_info": get_clinic_info,
    "check_availability": check_availability,
}


def tool_definitions(location_names: list[str]) -> list[dict[str, Any]]:
    """Describe the tools in the format the OpenAI Responses API expects."""
    # The location names come from the config file, because they're
    # clinic-specific. Offering them as an enum stops the model from
    # sending a location that doesn't exist.
    #
    # "strict": True makes the API guarantee the arguments match the
    # schema exactly. Strict mode requires every property to be listed in
    # "required", so an optional value is written as "string or null".
    get_clinic_info_definition = {
        "type": "function",
        "name": "get_clinic_info",
        "description": (
            "Look up a clinic fact: the opening hours, address or parking of"
            " one location, or the insurers the clinic accepts (the same at"
            " every location)."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": ["string", "null"],
                    "enum": location_names + [None],
                    "description": "The location. Use null only for insurers.",
                },
                "topic": {"type": "string", "enum": CLINIC_INFO_TOPICS},
            },
            "required": ["location", "topic"],
            "additionalProperties": False,
        },
    }
    check_availability_definition = {
        "type": "function",
        "name": "check_availability",
        "description": (
            "List open appointment slots at one location for an appointment"
            " type, between two dates (both included), and optionally between"
            " two start times. Also counts the open slots on each day searched,"
            " and says which of those days the location is closed."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "enum": location_names},
                "appointment_type": {"type": "string", "enum": APPOINTMENT_TYPES},
                "first_day": {"type": "string", "description": "YYYY-MM-DD"},
                "last_day": {"type": "string", "description": "YYYY-MM-DD, included"},
                "earliest_start": {
                    "type": ["string", "null"],
                    "description": (
                        "HH:MM, 24-hour: only slots starting at or after this."
                        " null for no limit."
                    ),
                },
                "latest_start": {
                    "type": ["string", "null"],
                    "description": (
                        "HH:MM, 24-hour: only slots starting at or before this."
                        " null for no limit."
                    ),
                },
            },
            "required": [
                "location",
                "appointment_type",
                "first_day",
                "last_day",
                "earliest_start",
                "latest_start",
            ],
            "additionalProperties": False,
        },
    }
    return [get_clinic_info_definition, check_availability_definition]


def run_tool(backend: httpx.Client, name: str, arguments_json: str) -> ToolResult:
    """Run the tool the model asked for. Never raises; failures become data."""
    tool_function = TOOL_FUNCTIONS.get(name)
    if tool_function is None:
        return tool_error(f"There is no tool named '{name}'.")

    # The model sends its arguments as a JSON string, not a dict.
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError:
        return tool_error("The tool arguments were not valid JSON.")
    if not isinstance(arguments, dict):
        return tool_error("The tool arguments must be a JSON object.")

    # A safety net: the tools already turn every failure they expect into
    # an error result, but a bug in a tool must not end the conversation.
    # Without strict mode, wrong argument names would also land here as a
    # TypeError. The chat loop prints the result, so the bug stays visible.
    try:
        return tool_function(backend, **arguments)
    except Exception as error:
        return tool_error(f"The tool failed unexpectedly: {type(error).__name__}: {error}")

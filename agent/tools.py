"""Tools the model can call, shared by the text agent and the voice agent.

Each tool calls the fake practice management system (backend/) over HTTP.
Every tool returns a dict the model can read, and failures come back as
{"ok": False, "error": "..."} instead of exceptions. The model can then
explain the problem to the caller, and the conversation keeps going.

Every tool takes the same first two arguments, the backend client and the
call's CallState, followed by the arguments the model chose. One calling
convention means run_tool needs no special cases.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
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

# When the requested dates have nothing open, how far before and after them
# to look for the closest open day, and how many of that option's times to
# send. Three is enough to offer the caller a choice.
ALTERNATIVE_SEARCH_DAYS = 14
ALTERNATIVE_SLOTS_PER_OPTION = 3

NOT_VERIFIED = (
    "This patient hasn't been verified on this call, so nothing was booked."
    " Ask for their full name, date of birth and phone number, and call"
    " verify_patient first."
)

ToolResult = dict[str, Any]


@dataclass
class CallState:
    """What the code, not the model, remembers about one call.

    The model's history can't hold this: the model writes part of the
    history itself, so it could claim a patient was verified. A fresh
    CallState is made for every call and is never shown to the model.
    """

    # Patients who passed verify_patient during this call. A set, because a
    # parent may verify two children (the seed has twins) on one call.
    # default_factory gives every CallState its own empty set; a plain
    # "= set()" default would be one set shared by every call.
    verified_patient_ids: set[int] = field(default_factory=set)


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
        field_name = problem["loc"][-1]
        messages.append(f"{field_name}: {problem['msg']}")
    return "Invalid request. " + "; ".join(messages)


def call_backend(
    backend: httpx.Client,
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> ToolResult:
    """Send one request to the backend; return {"ok": True, "data": ...} or an error result."""
    try:
        # json= sends the dict as a JSON request body, which is what the
        # POST endpoints read. GET requests leave it as None.
        response = backend.request(method, path, params=params, json=json_body)
    except httpx.HTTPError:
        # Connection refused, timeout, DNS failure: the backend isn't there.
        return tool_error("The clinic's scheduling system can't be reached right now.")

    # httpx doesn't raise on 4xx/5xx unless asked to, so check the status.
    if response.is_error:
        return tool_error(error_detail(response))
    return {"ok": True, "data": response.json()}


# --- Clinic information ---


def get_clinic_info(
    backend: httpx.Client, call_state: CallState, location: str | None, topic: str
) -> ToolResult:
    params = {"topic": topic}
    # Leaving location out of the URL is how the backend knows the
    # question is clinic-wide (insurers).
    if location is not None:
        params["location"] = location

    result = call_backend(backend, "GET", "/clinic-info", params)
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


# --- Availability ---


def find_location_id(backend: httpx.Client, location: str) -> ToolResult:
    """Turn a location name into the id /slots expects."""
    result = call_backend(backend, "GET", "/locations")
    if not result["ok"]:
        return result
    for known_location in result["data"]:
        if known_location["name"] == location:
            return {"ok": True, "location_id": known_location["id"]}
    return tool_error(f"There is no location named '{location}'.")


def fetch_open_slots(
    backend: httpx.Client,
    location_id: int,
    appointment_type: str,
    first_day: str,
    last_day: str,
) -> ToolResult:
    params = {
        "location": location_id,
        "type": appointment_type,
        "from": first_day,
        "to": last_day,
    }
    return call_backend(backend, "GET", "/slots", params)


def summarize_slots(slots: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep the first `limit` slots, with only the fields the model needs."""
    # Only what the model needs to offer and later book a slot. Fewer
    # fields means fewer tokens and a shorter debug line.
    summaries = []
    for slot in slots[:limit]:
        summaries.append(
            {
                "slot_id": slot["id"],
                "start_time": slot["start_time"],
                "provider": slot["provider_name"],
            }
        )
    return summaries


def day_of(slot: dict[str, Any]) -> str:
    # Slot times look like "2026-11-04T09:00", so the first ten characters
    # are the date. Comparing these "YYYY-MM-DD" strings compares the dates.
    return slot["start_time"][:10]


def slots_on_day(slots: list[dict[str, Any]], day: str) -> list[dict[str, Any]]:
    matching = []
    for slot in slots:
        if day_of(slot) == day:
            matching.append(slot)
    return matching


def closest_days_nearby(
    backend: httpx.Client,
    location_id: int,
    appointment_type: str,
    first_day: date,
    last_day: date,
) -> ToolResult:
    """The open slots on the nearest open day before and after the requested dates."""
    # One request for the weeks around the requested dates, then split in
    # Python. The backend already leaves out days in the past.
    search_start = first_day - timedelta(days=ALTERNATIVE_SEARCH_DAYS)
    search_end = last_day + timedelta(days=ALTERNATIVE_SEARCH_DAYS)
    result = fetch_open_slots(
        backend, location_id, appointment_type, search_start.isoformat(), search_end.isoformat()
    )
    if not result["ok"]:
        return result

    earlier_slots = []
    later_slots = []
    for slot in result["data"]:
        if day_of(slot) < first_day.isoformat():
            earlier_slots.append(slot)
        elif day_of(slot) > last_day.isoformat():
            later_slots.append(slot)

    # The slots come sorted by time, so the closest earlier day is the day
    # of the last earlier slot, and the closest later day is the day of the
    # first later slot. A side with nothing open is left out.
    nearby = {"ok": True}
    if earlier_slots:
        closest_earlier = slots_on_day(earlier_slots, day_of(earlier_slots[-1]))
        nearby["closest_earlier_day"] = summarize_slots(
            closest_earlier, ALTERNATIVE_SLOTS_PER_OPTION
        )
    if later_slots:
        closest_later = slots_on_day(later_slots, day_of(later_slots[0]))
        nearby["closest_later_day"] = summarize_slots(
            closest_later, ALTERNATIVE_SLOTS_PER_OPTION
        )
    return nearby


def other_locations_open(
    backend: httpx.Client,
    location: str,
    appointment_type: str,
    first_day: str,
    last_day: str,
) -> ToolResult:
    """The other locations with openings on the requested dates."""
    locations_result = call_backend(backend, "GET", "/locations")
    if not locations_result["ok"]:
        return locations_result

    open_locations = []
    for other in locations_result["data"]:
        if other["name"] == location:
            continue
        slots_result = fetch_open_slots(
            backend, other["id"], appointment_type, first_day, last_day
        )
        if not slots_result["ok"]:
            return slots_result
        if slots_result["data"]:
            open_locations.append(
                {
                    "location": other["name"],
                    "slots": summarize_slots(slots_result["data"], ALTERNATIVE_SLOTS_PER_OPTION),
                }
            )
    return {"ok": True, "other_locations": open_locations}


def find_alternatives(
    backend: httpx.Client,
    location: str,
    location_id: int,
    appointment_type: str,
    first_day: str,
    last_day: str,
) -> ToolResult:
    """The closest openings, for when the requested dates have none."""
    # Done in code rather than left to the model: finding these itself would
    # take several check_availability calls, and it tends to stop after one.
    # The /slots request for the same dates already succeeded, so these
    # strings are valid dates.
    nearby = closest_days_nearby(
        backend,
        location_id,
        appointment_type,
        date.fromisoformat(first_day),
        date.fromisoformat(last_day),
    )
    if not nearby["ok"]:
        return nearby
    elsewhere = other_locations_open(backend, location, appointment_type, first_day, last_day)
    if not elsewhere["ok"]:
        return elsewhere

    alternatives = {}
    if "closest_earlier_day" in nearby:
        alternatives["closest_earlier_day"] = nearby["closest_earlier_day"]
    if "closest_later_day" in nearby:
        alternatives["closest_later_day"] = nearby["closest_later_day"]
    alternatives["other_locations"] = elsewhere["other_locations"]
    return {"ok": True, "alternatives": alternatives}


def check_availability(
    backend: httpx.Client,
    call_state: CallState,
    location: str,
    appointment_type: str,
    first_day: str,
    last_day: str,
) -> ToolResult:
    location_result = find_location_id(backend, location)
    if not location_result["ok"]:
        return location_result
    location_id = location_result["location_id"]

    slots_result = fetch_open_slots(backend, location_id, appointment_type, first_day, last_day)
    if not slots_result["ok"]:
        return slots_result

    all_slots = slots_result["data"]
    result = {
        "ok": True,
        "location": location,
        "appointment_type": appointment_type,
        "open_slot_count": len(all_slots),
        "slots": summarize_slots(all_slots, MAX_SLOTS_RETURNED),
    }
    if len(all_slots) > MAX_SLOTS_RETURNED:
        result["note"] = (
            f"Only the earliest {MAX_SLOTS_RETURNED} are listed."
            " Search a narrower date range to see later times."
        )

    if not all_slots:
        alternatives_result = find_alternatives(
            backend, location, location_id, appointment_type, first_day, last_day
        )
        if not alternatives_result["ok"]:
            return alternatives_result
        result["alternatives"] = alternatives_result["alternatives"]
    return result


# --- Verification and booking ---


def verify_patient(
    backend: httpx.Client,
    call_state: CallState,
    name: str,
    date_of_birth: str,
    phone: str,
) -> ToolResult:
    body = {"name": name, "date_of_birth": date_of_birth, "phone": phone}
    result = call_backend(backend, "POST", "/patients/verify", json_body=body)
    if not result["ok"]:
        return result

    patient_id = result["data"]["patient_id"]
    # This is the only place a patient becomes verified, and only after the
    # backend matched all three details.
    call_state.verified_patient_ids.add(patient_id)
    return {"ok": True, "patient_id": patient_id}


def book_appointment(
    backend: httpx.Client,
    call_state: CallState,
    patient_id: int,
    slot_id: int,
    appointment_type: str,
) -> ToolResult:
    # Checked here in code, not only asked for in the prompt: a model can
    # skip a step in its instructions, or invent a patient_id. Then the
    # booking is refused before anything reaches the backend.
    if patient_id not in call_state.verified_patient_ids:
        return tool_error(NOT_VERIFIED)

    body = {"patient_id": patient_id, "slot_id": slot_id, "type": appointment_type}
    result = call_backend(backend, "POST", "/appointments", json_body=body)
    if not result["ok"]:
        return result

    # What the backend actually booked, so the agent confirms the real
    # appointment rather than what it remembers asking for.
    appointment = result["data"]
    return {
        "ok": True,
        "appointment_id": appointment["id"],
        "appointment_type": appointment["type"],
        "location": appointment["location_name"],
        "start_time": appointment["start_time"],
        "provider": appointment["provider_name"],
    }


# --- What the model sees, and how its requests are dispatched ---

TOOL_FUNCTIONS: dict[str, Callable[..., ToolResult]] = {
    "get_clinic_info": get_clinic_info,
    "check_availability": check_availability,
    "verify_patient": verify_patient,
    "book_appointment": book_appointment,
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
            " type, between two dates (both included). If none are open, the"
            " result lists the closest alternatives."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "enum": location_names},
                "appointment_type": {"type": "string", "enum": APPOINTMENT_TYPES},
                "first_day": {"type": "string", "description": "YYYY-MM-DD"},
                "last_day": {"type": "string", "description": "YYYY-MM-DD, included"},
            },
            "required": ["location", "appointment_type", "first_day", "last_day"],
            "additionalProperties": False,
        },
    }
    verify_patient_definition = {
        "type": "function",
        "name": "verify_patient",
        "description": (
            "Check that the caller is a patient on file. The name, date of"
            " birth and phone number must all match. Required before booking."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "First and last name"},
                "date_of_birth": {"type": "string", "description": "YYYY-MM-DD"},
                "phone": {"type": "string", "description": "As the caller says it"},
            },
            "required": ["name", "date_of_birth", "phone"],
            "additionalProperties": False,
        },
    }
    book_appointment_definition = {
        "type": "function",
        "name": "book_appointment",
        "description": (
            "Book an open slot for a patient verified on this call. Only call"
            " it after the caller has confirmed the day, time and location."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer", "description": "From verify_patient"},
                "slot_id": {"type": "integer", "description": "From check_availability"},
                "appointment_type": {"type": "string", "enum": APPOINTMENT_TYPES},
            },
            "required": ["patient_id", "slot_id", "appointment_type"],
            "additionalProperties": False,
        },
    }
    return [
        get_clinic_info_definition,
        check_availability_definition,
        verify_patient_definition,
        book_appointment_definition,
    ]


def run_tool(
    backend: httpx.Client, call_state: CallState, name: str, arguments_json: str
) -> ToolResult:
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
        return tool_function(backend, call_state, **arguments)
    except Exception as error:
        return tool_error(f"The tool failed unexpectedly: {type(error).__name__}: {error}")

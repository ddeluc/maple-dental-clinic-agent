"""Tools the model can call, shared by the text agent and the voice agent.

Each tool calls the fake practice management system (backend/) over HTTP.
Every tool returns a dict the model can read, and failures come back as
{"ok": False, "error": "..."} instead of exceptions. The model can then
explain the problem to the caller, and the conversation keeps going.
"""

import json
from collections.abc import Callable
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

ToolResult = dict[str, Any]


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


def check_availability(
    backend: httpx.Client,
    location: str,
    appointment_type: str,
    first_day: str,
    last_day: str,
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
    slots_result = get_from_backend(backend, "/slots", params)
    if not slots_result["ok"]:
        return slots_result

    all_slots = slots_result["data"]
    # Only what the model needs to offer and later book a slot. Fewer
    # fields means fewer tokens and a shorter debug line.
    slots = []
    for slot in all_slots[:MAX_SLOTS_RETURNED]:
        slots.append(
            {
                "slot_id": slot["id"],
                "start_time": slot["start_time"],
                "provider": slot["provider_name"],
            }
        )

    result = {
        "ok": True,
        "location": location,
        "appointment_type": appointment_type,
        "open_slot_count": len(all_slots),
        "slots": slots,
    }
    if len(all_slots) > MAX_SLOTS_RETURNED:
        result["note"] = (
            f"Only the earliest {MAX_SLOTS_RETURNED} are listed."
            " Search a narrower date range to see later times."
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
            " type, between two dates (both included)."
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

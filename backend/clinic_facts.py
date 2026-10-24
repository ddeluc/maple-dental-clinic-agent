"""Clinic facts (locations, opening hours, parking, insurers) from config/.

The seed script and the /clinic-info endpoint both read them from here, so
the slots in the database and the hours the agent quotes can't disagree.
"""

from datetime import date, time
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "maple_dental.yaml"

# Indexed by date.weekday(), which numbers Monday as 0. Spelled out instead
# of using calendar.day_name, whose names follow the computer's language
# settings and wouldn't match the keys in the YAML on a French system.
DAY_NAMES = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
]
CLOSED = "closed"
LOCATION_KEYS = ("address", "parking", "hours")


def parse_hours(text: str) -> tuple[time, time] | None:
    """Turn "08:00-17:00" into (open, close), or "closed" into None."""
    if text == CLOSED:
        return None
    if not isinstance(text, str) or text.count("-") != 1:
        raise ValueError(f"Hours must look like '08:00-17:00' or 'closed', not {text!r}.")
    opens_text, closes_text = text.split("-")
    return time.fromisoformat(opens_text), time.fromisoformat(closes_text)


def check_location(name: str, location: Any) -> None:
    """Raise ValueError if one location's facts are missing or malformed."""
    if not isinstance(location, dict):
        raise ValueError(f"Location '{name}' must contain address, parking and hours.")
    for key in LOCATION_KEYS:
        if key not in location:
            raise ValueError(f"Location '{name}' needs '{key}'.")
    hours = location["hours"]
    for day_name in DAY_NAMES:
        if not isinstance(hours, dict) or day_name not in hours:
            raise ValueError(f"Location '{name}' needs hours for {day_name}.")
        # Parsing here, at startup, turns a typo like "8h-17h" into a clear
        # error now instead of a crash halfway through seeding.
        parse_hours(hours[day_name])


def load_clinic_facts(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Return {"locations": {name: facts}, "insurers": [names]} from the config."""
    # The agent has its own loader for the same file (agent/config.py). The
    # backend reads it separately so it never imports agent code: it stands
    # in for a separate system the agent only talks to over HTTP.
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    locations = config.get("locations")
    if not isinstance(locations, dict) or not locations:
        raise ValueError(f"{path} needs a 'locations' section.")
    for name, location in locations.items():
        check_location(name, location)

    insurers = config.get("insurers")
    if not isinstance(insurers, list):
        raise ValueError(f"{path} needs an 'insurers' list.")

    return {"locations": locations, "insurers": insurers}


def opening_hours(
    facts: dict[str, Any], location_name: str, day: date
) -> tuple[time, time] | None:
    """Return (open, close) for a location on a day, or None if closed."""
    hours = facts["locations"][location_name]["hours"]
    return parse_hours(hours[DAY_NAMES[day.weekday()]])

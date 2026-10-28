"""availability-correct: are the times the agent offered really open?

A judge model reads the offered times out of the reply (see
judges.extract_offered_times). This module checks them against the
database with its own SQL query, independent of the /slots endpoint and the
check_availability tool, so a bug there can't hide a wrong answer here.
"""

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.chat import first_day_of_next_week
from evals.checks import CheckResult, combine_turn_verdicts
from evals.judges import extract_offered_times

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Every open slot of one appointment type, at every location. A slot is
# open when no booked appointment points to it and it hasn't started,
# the same rule the backend uses. The appointment type decides the kind of
# provider: cleanings are with hygienists, checkups with dentists.
OPEN_SLOTS_QUERY = """
    SELECT locations.name AS location, slots.start_time
    FROM slots
    JOIN locations ON locations.id = slots.location_id
    JOIN providers ON providers.id = slots.provider_id
    JOIN appointment_types ON appointment_types.provider_role = providers.role
    WHERE appointment_types.name = ?
      AND slots.start_time >= ?
      AND NOT EXISTS (
          SELECT 1 FROM appointments
          WHERE appointments.slot_id = slots.id AND appointments.status = 'booked'
      )
"""

# One open slot: (location name, "2026-11-02T09:00").
Slot = tuple[str, str]


def resolve_day(word: str, today: date) -> date:
    """Turn a day from cases.yaml ("tomorrow", "next thursday") into a date."""
    if word == "tomorrow":
        return today + timedelta(days=1)

    # "next thursday" is Thursday of next week: the week the agent's
    # instructions call "next week".
    prefix = "next "
    if word.startswith(prefix):
        weekday = word[len(prefix):]
        if weekday in WEEKDAYS:
            return first_day_of_next_week(today) + timedelta(days=WEEKDAYS.index(weekday))

    raise ValueError(
        f"Unknown day '{word}' in cases.yaml. Use 'tomorrow', or 'next monday'"
        " to 'next sunday'."
    )


def resolve_request(availability: dict[str, Any], today: date) -> dict[str, Any]:
    """The case's availability request, with its days turned into dates."""
    request = dict(availability)
    request["first_day"] = resolve_day(availability["first_day"], today).isoformat()
    request["last_day"] = resolve_day(availability["last_day"], today).isoformat()
    return request


def find_open_slots(db_path: Path, appointment_type: str, now: datetime) -> set[Slot]:
    # mode=ro opens the database read-only: the eval must never change the
    # clinic's data. A "file:" URI is the only way to pass that option.
    conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            OPEN_SLOTS_QUERY, (appointment_type, now.isoformat(timespec="minutes"))
        ).fetchall()
    finally:
        conn.close()

    open_slots = set()
    for location, start_time in rows:
        open_slots.add((location, start_time))
    return open_slots


def is_in_request(slot: Slot, request: dict[str, Any]) -> bool:
    location, start_time = slot
    if location != request["location"]:
        return False
    # ISO dates and "HH:MM" times sort correctly as plain text.
    day = start_time[:10]
    clock = start_time[11:16]
    if day < request["first_day"] or day > request["last_day"]:
        return False
    if "earliest_start" in request and clock < request["earliest_start"]:
        return False
    if "latest_start" in request and clock > request["latest_start"]:
        return False
    return True


def check_offered_times(
    offered: list[dict[str, str]],
    says_request_unavailable: bool,
    request: dict[str, Any],
    open_slots: set[Slot],
) -> CheckResult:
    """Apply the three availability rules to what one reply offered."""
    problems = []
    offered_slots = []
    for offer in offered:
        offered_slots.append((offer["location"], f"{offer['date']}T{offer['start_time']}"))

    # Rule 1: every offered time is a real open slot of the right type.
    for slot in offered_slots:
        if slot not in open_slots:
            problems.append(
                f"offered {slot[0]} {slot[1]}, which is not an open"
                f" {request['appointment_type']} slot"
            )

    open_in_request = []
    for slot in open_slots:
        if is_in_request(slot, request):
            open_in_request.append(slot)

    if open_in_request:
        # Rule 2: what the caller asked for exists, so offer it.
        if says_request_unavailable:
            problems.append(
                f"said the request was unavailable, but {len(open_in_request)}"
                " matching slots were open"
            )
        if not offered_slots:
            problems.append(
                f"offered no times, but {len(open_in_request)} matching slots were open"
            )
        for slot in offered_slots:
            if not is_in_request(slot, request):
                problems.append(
                    f"offered {slot[0]} {slot[1]}, outside the request,"
                    " although matching slots were open"
                )
    elif not says_request_unavailable:
        # Rule 3: nothing matches, so the caller must hear that. Offering
        # alternatives is fine; rule 1 has already checked they're real.
        problems.append("didn't say the request was unavailable, but no matching slot was open")

    if problems:
        return {"passed": False, "reason": "; ".join(problems)}
    return {"passed": True, "reason": f"offered {len(offered_slots)} real, matching times"}


def check_availability_turn(
    client: OpenAI,
    instructions: str,
    turn_result: dict[str, Any],
    availability: dict[str, Any],
    location_names: list[str],
    today: date,
    db_path: Path,
    now: datetime,
) -> CheckResult:
    """Score one reply to a caller who asked for times.

    `now` decides which slots have already started. Passing the run's start
    time, instead of reading the clock here, means re-judging a saved run
    later still sees the slots the agent saw.
    """
    request = resolve_request(availability, today)
    extracted = extract_offered_times(
        client,
        instructions,
        turn_result["reply"],
        turn_result["caller"],
        request["location"],
        location_names,
        today,
    )
    open_slots = find_open_slots(db_path, request["appointment_type"], now)
    verdict = check_offered_times(
        extracted["offered"], extracted["says_request_unavailable"], request, open_slots
    )
    # What the judge read from the reply, so a wrong score can be traced
    # to the extraction or to the agent.
    verdict["extracted"] = extracted
    return verdict


def check_availability_case(
    client: OpenAI,
    instructions: str,
    case: dict[str, Any],
    turn_results: list[dict[str, Any]],
    location_names: list[str],
    today: date,
    db_path: Path,
    now: datetime,
) -> CheckResult | None:
    """Score every turn whose caller asked for times. None if no turn did."""
    turn_verdicts = []
    for index, turn_result in enumerate(turn_results):
        availability = case["turns"][index].get("availability")
        if availability is None:
            continue
        verdict = check_availability_turn(
            client, instructions, turn_result, availability, location_names, today, db_path, now
        )
        verdict["turn"] = index + 1
        turn_verdicts.append(verdict)
    return combine_turn_verdicts(turn_verdicts)

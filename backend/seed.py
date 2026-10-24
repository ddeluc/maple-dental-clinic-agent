"""Create backend/clinic.db from scratch and fill it with fake clinic data.

Run with: uv run python backend/seed.py

Every run deletes the old database first, so the result is always the same
clean starting point. All patient data is fictional.
"""

import random
import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
DB_PATH = BACKEND_DIR / "clinic.db"
SCHEMA_PATH = BACKEND_DIR / "schema.sql"

# A fixed seed makes the "random" patients and appointments identical on
# every run, so tests and evals can rely on the same data.
RANDOM_SEED = 42

SLOT_DAYS = 28  # 4 weeks of open slots, starting today

LOCATIONS = ["Plateau", "Verdun", "Saint-Laurent"]

# Hours live here for now, as agreed in Step 1.1. They should move to
# config/ once get_clinic_info needs them too.
WEEKDAY_HOURS = (time(8, 0), time(17, 0))
SATURDAY_HOURS = (time(9, 0), time(13, 0))
LOCATIONS_OPEN_SATURDAY = ["Plateau"]
SATURDAY = 5  # date.weekday() numbers Monday as 0 and Sunday as 6

APPOINTMENT_TYPES = [
    # (name, provider_role, duration_minutes)
    ("cleaning", "hygienist", 60),
    ("checkup", "dentist", 30),
]

PROVIDERS = [
    # (name, role, location)
    ("Sophie Gagnon", "dentist", "Plateau"),
    ("Marc Lefebvre", "dentist", "Plateau"),
    ("Amira Haddad", "dentist", "Verdun"),
    ("Daniel Cohen", "dentist", "Verdun"),
    ("Julie Nguyen", "dentist", "Saint-Laurent"),
    ("Patrick O'Brien", "dentist", "Saint-Laurent"),
    ("Chloé Roy", "hygienist", "Plateau"),
    ("Maya Singh", "hygienist", "Plateau"),
    ("Isabelle Côté", "hygienist", "Verdun"),
    ("Kevin Martin", "hygienist", "Verdun"),
    ("Nadia Benali", "hygienist", "Saint-Laurent"),
]

# Hand-written patients that make verification hard on purpose, as the
# project definition asks: duplicate names, close dates of birth, accents,
# and phone numbers typed in different formats. 555-0100 to 555-0199 are
# reserved for fiction, so none of these can be a real person's number.
MESSY_PATIENTS = [
    # (first_name, last_name, date_of_birth, phone)
    # Same name, birthdays with day and month swapped
    ("Marie", "Tremblay", "1985-03-04", "514-555-0100"),
    ("Marie", "Tremblay", "1985-04-03", "(514) 555-0101"),
    # Same name with and without accents, so "Helene" matches both
    ("Hélène", "Bouchard", "1972-06-08", "514.555.0102"),
    ("Helene", "Bouchard", "1972-06-18", "5145550103"),
    # Hyphen and accents in both names
    ("Jean-François", "Côté-Lavoie", "1990-11-23", "+1 514 555 0104"),
    # Twins sharing a household phone number
    ("Liam", "Wong", "2012-09-15", "514-555-0105"),
    ("Noah", "Wong", "2012-09-15", "514-555-0105"),
    # Same name and birthday, different people
    ("Michael", "Smith", "1968-01-30", "514 555 0106"),
    ("Michael", "Smith", "1968-01-30", "438-555-0107"),
]

GENERATED_PATIENT_COUNT = 50 - len(MESSY_PATIENTS)
FIRST_PHONE_SUFFIX_FOR_GENERATED = 110  # after the messy patients' numbers

FIRST_NAMES = [
    "Émilie", "Olivier", "Camille", "Gabriel", "Léa", "Thomas", "Zoé",
    "Samuel", "Chloé", "William", "Sarah", "Alexandre", "Ana", "David",
    "Fatima", "Lucas", "Mélanie", "Ethan", "Yasmine", "Nathan",
]

LAST_NAMES = [
    "Gagnon", "Roy", "Bélanger", "Morin", "Lavoie", "Fortin", "Gauthier",
    "Pelletier", "Ouellet", "Leblanc", "Chen", "Patel", "Rossi", "Garcia",
    "Kowalski", "Dubois", "Bergeron", "Nadeau", "Thériault", "Murphy",
]

BOOKED_APPOINTMENT_COUNT = 15
CANCELLED_APPOINTMENTS = [
    "Patient is travelling",
    "Scheduling conflict at work",
    "Feeling unwell",
]


def opening_hours(location_name: str, day: date) -> tuple[time, time] | None:
    """Return (open, close) for a location on a day, or None if closed."""
    if day.weekday() < SATURDAY:
        return WEEKDAY_HOURS
    if day.weekday() == SATURDAY and location_name in LOCATIONS_OPEN_SATURDAY:
        return SATURDAY_HOURS
    return None


def slot_times(
    day: date, opens: time, closes: time, duration_minutes: int
) -> list[tuple[datetime, datetime]]:
    """Split one day's opening hours into back-to-back slots."""
    length = timedelta(minutes=duration_minutes)
    start = datetime.combine(day, opens)
    closing_time = datetime.combine(day, closes)

    times = []
    # Stop when a slot would run past closing, not just start before it.
    while start + length <= closing_time:
        times.append((start, start + length))
        start = start + length
    return times


def create_tables(conn: sqlite3.Connection) -> None:
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema)


def insert_locations(conn: sqlite3.Connection) -> dict[str, int]:
    """Insert locations and return a {name: id} lookup."""
    location_ids = {}
    for name in LOCATIONS:
        cursor = conn.execute("INSERT INTO locations (name) VALUES (?)", (name,))
        location_ids[name] = cursor.lastrowid
    return location_ids


def insert_appointment_types(conn: sqlite3.Connection) -> dict[str, int]:
    """Insert appointment types and return a {provider_role: type_id} lookup."""
    type_id_by_role = {}
    for name, provider_role, duration_minutes in APPOINTMENT_TYPES:
        cursor = conn.execute(
            "INSERT INTO appointment_types (name, provider_role, duration_minutes)"
            " VALUES (?, ?, ?)",
            (name, provider_role, duration_minutes),
        )
        type_id_by_role[provider_role] = cursor.lastrowid
    return type_id_by_role


def insert_providers(
    conn: sqlite3.Connection, location_ids: dict[str, int]
) -> list[tuple[int, str, str]]:
    """Insert providers and return (provider_id, role, location_name) for each."""
    providers = []
    for name, role, location_name in PROVIDERS:
        cursor = conn.execute(
            "INSERT INTO providers (name, role, location_id) VALUES (?, ?, ?)",
            (name, role, location_ids[location_name]),
        )
        providers.append((cursor.lastrowid, role, location_name))
    return providers


def random_date_of_birth(rng: random.Random) -> str:
    earliest = date(1940, 1, 1)
    latest = date(2015, 12, 31)
    days_in_range = (latest - earliest).days
    birthday = earliest + timedelta(days=rng.randint(0, days_in_range))
    return birthday.isoformat()


def insert_patients(conn: sqlite3.Connection, rng: random.Random) -> list[int]:
    """Insert the messy patients plus generated ones; return all patient ids."""
    patients = list(MESSY_PATIENTS)

    for index in range(GENERATED_PATIENT_COUNT):
        first_name = rng.choice(FIRST_NAMES)
        last_name = rng.choice(LAST_NAMES)
        date_of_birth = random_date_of_birth(rng)
        # The index makes every generated phone number unique.
        phone_suffix = FIRST_PHONE_SUFFIX_FOR_GENERATED + index
        phone = f"514-555-0{phone_suffix}"
        patients.append((first_name, last_name, date_of_birth, phone))

    patient_ids = []
    for first_name, last_name, date_of_birth, phone in patients:
        cursor = conn.execute(
            "INSERT INTO patients (first_name, last_name, date_of_birth, phone)"
            " VALUES (?, ?, ?, ?)",
            (first_name, last_name, date_of_birth, phone),
        )
        patient_ids.append(cursor.lastrowid)
    return patient_ids


def insert_slots(
    conn: sqlite3.Connection,
    providers: list[tuple[int, str, str]],
    location_ids: dict[str, int],
    first_day: date,
) -> None:
    """Create every provider's slots for SLOT_DAYS days, within opening hours."""
    duration_by_role = {}
    for _name, provider_role, duration_minutes in APPOINTMENT_TYPES:
        duration_by_role[provider_role] = duration_minutes

    for day_offset in range(SLOT_DAYS):
        day = first_day + timedelta(days=day_offset)
        for provider_id, role, location_name in providers:
            hours = opening_hours(location_name, day)
            if hours is None:
                continue
            opens, closes = hours
            for start, end in slot_times(day, opens, closes, duration_by_role[role]):
                conn.execute(
                    "INSERT INTO slots (provider_id, location_id, start_time, end_time)"
                    " VALUES (?, ?, ?, ?)",
                    (
                        provider_id,
                        location_ids[location_name],
                        start.isoformat(timespec="minutes"),
                        end.isoformat(timespec="minutes"),
                    ),
                )


def insert_appointments(
    conn: sqlite3.Connection,
    rng: random.Random,
    patient_ids: list[int],
    type_id_by_role: dict[str, int],
) -> None:
    """Book some slots, and add a few cancelled appointments.

    Later steps need existing appointments to reschedule and cancel.
    """
    slot_rows = conn.execute(
        "SELECT slots.id, providers.role FROM slots"
        " JOIN providers ON providers.id = slots.provider_id"
        " ORDER BY slots.id"
    ).fetchall()

    appointment_count = BOOKED_APPOINTMENT_COUNT + len(CANCELLED_APPOINTMENTS)
    # sample() picks without repeats, so no patient or slot is used twice.
    chosen_slots = rng.sample(slot_rows, appointment_count)
    chosen_patients = rng.sample(patient_ids, appointment_count)

    # The first ones are booked; the rest are cancelled with a reason.
    statuses_and_reasons = [("booked", None)] * BOOKED_APPOINTMENT_COUNT
    for reason in CANCELLED_APPOINTMENTS:
        statuses_and_reasons.append(("cancelled", reason))

    created_at = datetime.now().isoformat(timespec="seconds")
    for index in range(appointment_count):
        slot_id, role = chosen_slots[index]
        status, cancel_reason = statuses_and_reasons[index]
        conn.execute(
            "INSERT INTO appointments"
            " (patient_id, slot_id, appointment_type_id, status, cancel_reason, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                chosen_patients[index],
                slot_id,
                type_id_by_role[role],
                status,
                cancel_reason,
                created_at,
            ),
        )


def seed(db_path: Path = DB_PATH, first_day: date | None = None) -> None:
    """Delete the database at db_path and rebuild it with seed data."""
    if first_day is None:
        first_day = date.today()

    # Starting from an empty file is what makes every run identical.
    db_path.unlink(missing_ok=True)

    rng = random.Random(RANDOM_SEED)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        create_tables(conn)
        location_ids = insert_locations(conn)
        type_id_by_role = insert_appointment_types(conn)
        providers = insert_providers(conn, location_ids)
        patient_ids = insert_patients(conn, rng)
        insert_slots(conn, providers, location_ids, first_day)
        insert_appointments(conn, rng, patient_ids, type_id_by_role)
        conn.commit()
    finally:
        conn.close()


def print_row_counts(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        tables = [
            "locations", "providers", "patients",
            "appointment_types", "slots", "appointments",
        ]
        for table in tables:
            # Table names can't be "?" parameters; these come from the list above.
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table:<18} {count}")
    finally:
        conn.close()


if __name__ == "__main__":
    seed()
    print(f"Created {DB_PATH}")
    print_row_counts(DB_PATH)

-- Schema for the fake practice management system.
-- All times are clinic local time (America/Montreal), stored as ISO 8601 text
-- ("2026-10-26T09:00"). SQLite has no datetime type, and this fixed,
-- zero-padded format sorts correctly as plain text.

CREATE TABLE locations (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE providers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('dentist', 'hygienist')),
    location_id INTEGER NOT NULL REFERENCES locations (id)
);

CREATE TABLE patients (
    id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    date_of_birth TEXT NOT NULL,  -- YYYY-MM-DD
    -- Kept exactly as entered (formats vary on purpose); verification
    -- compares digits only.
    phone TEXT NOT NULL
);

CREATE TABLE appointment_types (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    provider_role TEXT NOT NULL CHECK (provider_role IN ('dentist', 'hygienist')),
    duration_minutes INTEGER NOT NULL
);

CREATE TABLE slots (
    id INTEGER PRIMARY KEY,
    provider_id INTEGER NOT NULL REFERENCES providers (id),
    location_id INTEGER NOT NULL REFERENCES locations (id),
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    UNIQUE (provider_id, start_time)
);

CREATE TABLE appointments (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients (id),
    slot_id INTEGER NOT NULL REFERENCES slots (id),
    appointment_type_id INTEGER NOT NULL REFERENCES appointment_types (id),
    status TEXT NOT NULL CHECK (status IN ('booked', 'cancelled')),
    cancel_reason TEXT,
    created_at TEXT NOT NULL
);

-- A slot is open when no booked appointment points to it. This partial index
-- lets a slot keep any number of cancelled appointments but at most one
-- booked one, so the database itself rejects a double booking even if the
-- Python code has a bug.
CREATE UNIQUE INDEX one_booking_per_slot ON appointments (slot_id)
    WHERE status = 'booked';

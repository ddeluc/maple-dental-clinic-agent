from datetime import date, time
from pathlib import Path

import pytest

from backend.clinic_facts import load_clinic_facts, opening_hours, parse_hours

# 2026-10-31 is a Saturday.
SATURDAY = date(2026, 10, 31)


def test_parse_hours_reads_open_and_close_times():
    assert parse_hours("08:00-17:00") == (time(8, 0), time(17, 0))


def test_parse_hours_closed_is_none():
    assert parse_hours("closed") is None


def test_parse_hours_rejects_other_formats():
    with pytest.raises(ValueError):
        parse_hours("8h-17h")


def test_only_plateau_opens_on_saturday():
    facts = load_clinic_facts()
    assert opening_hours(facts, "Plateau", SATURDAY) == (time(9, 0), time(13, 0))
    assert opening_hours(facts, "Verdun", SATURDAY) is None
    assert opening_hours(facts, "Saint-Laurent", SATURDAY) is None


def test_load_clinic_facts_rejects_missing_day(tmp_path: Path):
    config_path = tmp_path / "clinic.yaml"
    config_path.write_text(
        "locations:\n"
        "  Verdun:\n"
        "    address: 1 Fake Street\n"
        "    parking: Street parking\n"
        "    hours:\n"
        "      monday: '08:00-17:00'\n"
        "insurers: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tuesday"):
        load_clinic_facts(config_path)

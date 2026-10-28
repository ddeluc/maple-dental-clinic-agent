import json
from datetime import date
from pathlib import Path

import pytest

from backend.seed import seed
from evals.calibration import (
    agreement_by_score,
    build_label_sheet,
    compare_sheet,
    judged_scores_for_turn,
    load_synthetic,
    meets_target,
    parse_item_id,
    write_label_sheet,
)
from evals.judges import load_judge_prompts
from evals.run import load_cases
from tests.fake_judge import FakeJudgeClient

GREETING = "Bonjour, hi!"
LOCATIONS = ["Plateau", "Verdun", "Saint-Laurent"]
JUDGE_REASON = "SECRET JUDGE REASONING"

CASE = {
    "id": "en-test",
    "language": "english",
    "turns": [
        {
            "caller": "A cleaning at Plateau next Monday?",
            "expected_behaviour": "Offers two or three times.",
            "availability": {
                "location": "Plateau",
                "appointment_type": "cleaning",
                "first_day": "next monday",
                "last_day": "next monday",
            },
        },
        {"caller": "Thanks!", "expected_behaviour": "Says goodbye."},
    ],
}


def verdict(turn: int) -> dict:
    return {"passed": True, "reason": JUDGE_REASON, "turn": turn}


RUN_RECORD = {
    "run_id": "run-20261027-150000",
    "started_at": "2026-10-27T15:00:00",
    "today": "2026-10-27",
    "cases": [
        {
            "id": "en-test",
            "turns": [
                {
                    "caller": "A cleaning at Plateau next Monday?",
                    "reply": "Monday at 8 or 9 a.m.?",
                    "language": "english",
                    "tool_calls": [
                        {"name": "check_availability", "arguments": {}, "result": {"ok": True}}
                    ],
                },
                {"caller": "Thanks!", "reply": "Goodbye!", "language": "english", "tool_calls": []},
            ],
            "scores": {
                # A code check: no per-turn verdicts, so never on the sheet.
                "language-detected": {"passed": True, "reason": "detected english"},
                "grounded": {"passed": True, "reason": "", "turns": [verdict(1), verdict(2)]},
                "availability-correct": {"passed": True, "reason": "", "turns": [verdict(1)]},
            },
        }
    ],
}

SYNTHETIC = [{"case_id": "en-test", "turn": 1, "reply": "Here: 08:00, 09:00, 10:00, 11:00."}]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "clinic.db"
    seed(path, first_day=date(2026, 10, 26))
    return path


def build(db_path: Path) -> dict:
    return build_label_sheet(
        RUN_RECORD, Path("evals/results/run.json"), [CASE], SYNTHETIC, GREETING, db_path
    )


def test_judged_scores_for_turn_reads_the_runs_per_turn_verdicts():
    case_record = RUN_RECORD["cases"][0]
    assert judged_scores_for_turn(case_record, 1) == ["grounded", "availability-correct"]
    assert judged_scores_for_turn(case_record, 2) == ["grounded"]


def test_sheet_has_each_real_reply_then_its_hand_written_alternative(db_path):
    items = build(db_path)["items"]

    assert [item["item_id"] for item in items] == [
        "en-test/turn-1",
        "en-test/turn-1/alt-1",
        "en-test/turn-2",
    ]
    assert items[0]["reply"] == "Monday at 8 or 9 a.m.?"
    assert items[1]["reply"] == "Here: 08:00, 09:00, 10:00, 11:00."
    # Replacing a reply for the alternative leaves the run itself unchanged.
    assert RUN_RECORD["cases"][0]["turns"][0]["reply"] == "Monday at 8 or 9 a.m.?"


def test_sheet_asks_for_the_judged_labels_and_hides_the_verdicts(db_path):
    sheet = build(db_path)

    assert sheet["items"][0]["labels"] == {"grounded": None, "availability-correct": None}
    assert sheet["items"][2]["labels"] == {"grounded": None}
    # Labels must be blind: no judge reasoning anywhere on the sheet.
    assert JUDGE_REASON not in json.dumps(sheet)


def test_sheet_shows_the_context_needed_to_label(db_path):
    item = build(db_path)["items"][2]

    assert item["required_language"] == "english"
    assert item["conversation"] == [
        f"Agent: {GREETING}",
        "Caller: A cleaning at Plateau next Monday?",
        "Agent: Monday at 8 or 9 a.m.?",
        "Caller: Thanks!",
    ]
    # Tool results from earlier turns count for grounded.
    assert item["tool_results_so_far"][0]["name"] == "check_availability"
    assert item["expected_behaviour"] == "Says goodbye."


def test_availability_items_list_the_open_slots_that_match(db_path):
    availability = build(db_path)["items"][0]["availability"]

    # From Tuesday Oct 27, "next monday" is Nov 2.
    assert availability["request"]["first_day"] == "2026-11-02"
    assert availability["open_matching_count"] > 0
    for start_time in availability["open_matching_slots"]:
        assert start_time.startswith("2026-11-02T")


def test_write_label_sheet_never_overwrites_labels(tmp_path):
    path = tmp_path / "labels.json"
    write_label_sheet({"items": []}, path)

    with pytest.raises(FileExistsError):
        write_label_sheet({"items": []}, path)


def test_every_hand_written_reply_points_at_a_real_turn():
    turn_counts = {}
    for case in load_cases():
        turn_counts[case["id"]] = len(case["turns"])

    for entry in load_synthetic():
        assert entry["case_id"] in turn_counts, entry["case_id"]
        assert 1 <= entry["turn"] <= turn_counts[entry["case_id"]]
        assert entry["reply"].strip()
        assert entry["meant_to_test"]


# --- compare ---


def sheet_item(item_id: str, reply: str, labels: dict, notes: str = "") -> dict:
    return {"item_id": item_id, "reply": reply, "labels": labels, "notes": notes}


def compare(client: FakeJudgeClient, items: list[dict], db_path: Path) -> list[dict]:
    return compare_sheet(
        client, load_judge_prompts(), {"items": items}, RUN_RECORD, [CASE], GREETING,
        LOCATIONS, db_path,
    )


def test_parse_item_id_reads_the_case_and_the_turn():
    assert parse_item_id("en-test/turn-2") == ("en-test", 2)
    assert parse_item_id("en-test/turn-2/alt-1") == ("en-test", 2)


def test_compare_judges_the_reply_that_was_labelled(db_path):
    item = sheet_item(
        "en-test/turn-2/alt-1", "Goodbye! Parking is $5.", {"grounded": "fail"}, "Invented fee."
    )
    client = FakeJudgeClient([{"reasoning": "no parking fee in the tools", "verdict": "fail"}])

    comparisons = compare(client, [item], db_path)

    assert comparisons == [
        {
            "item_id": "en-test/turn-2/alt-1",
            "score": "grounded",
            "label": "fail",
            "judge": "fail",
            "judge_reason": "no parking fee in the tools",
            "notes": "Invented fee.",
        }
    ]
    call = client.calls[0]
    assert call["instructions"] == load_judge_prompts()["grounded"]
    # The hand-written reply is judged, after the run's real earlier turn.
    assert call["input"].endswith("Reply to check:\nGoodbye! Parking is $5.")
    assert "Agent: Monday at 8 or 9 a.m.?" in call["input"]
    assert RUN_RECORD["cases"][0]["turns"][1]["reply"] == "Goodbye!"


def test_compare_checks_offered_times_against_the_database(db_path):
    item = sheet_item("en-test/turn-1", "Monday at 9:15?", {"availability-correct": "fail"})
    extracted = {
        "offered": [{"date": "2026-11-02", "start_time": "09:15", "location": "Plateau"}],
        "says_request_unavailable": False,
    }

    comparisons = compare(FakeJudgeClient([extracted]), [item], db_path)

    assert comparisons[0]["judge"] == "fail"
    assert "not an open cleaning slot" in comparisons[0]["judge_reason"]
    # Kept so a disagreement can be traced to the extraction or the check.
    assert comparisons[0]["extracted"] == extracted


def comparison(score: str, label: str, judge: str) -> dict:
    return {"score": score, "label": label, "judge": judge}


def test_agreement_counts_missed_failures_and_false_alarms():
    agreement = agreement_by_score(
        [
            comparison("grounded", "pass", "pass"),
            comparison("grounded", "fail", "pass"),  # missed failure
            comparison("grounded", "pass", "fail"),  # false alarm
            comparison("phone-style", "fail", "fail"),
        ]
    )

    assert agreement == {
        "grounded": {"labels": 3, "agreed": 1, "missed_failures": 1, "false_alarms": 1},
        "phone-style": {"labels": 1, "agreed": 1, "missed_failures": 0, "false_alarms": 0},
    }


def test_meets_target_needs_85_percent_and_no_missed_language_or_grounded_failures():
    # 9 of 10 agree: above the target, but the one miss is a missed failure.
    counts = {"labels": 10, "agreed": 9, "missed_failures": 1, "false_alarms": 0}
    assert meets_target("phone-style", counts)
    assert not meets_target("grounded", counts)

    too_few = {"labels": 10, "agreed": 8, "missed_failures": 0, "false_alarms": 2}
    assert not meets_target("phone-style", too_few)

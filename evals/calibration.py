"""Judge calibration: check the judges against a person's labels.

Build a label sheet from a saved run plus the hand-written replies in
calibration/synthetic.yaml:

    uv run python -m evals.calibration sheet evals/results/<run>.json

Then replace every null under "labels" in calibration/labels.json with
"pass" or "fail". The sheet doesn't show the judges' verdicts, so they
can't steer the labels.

Once labelled, have the judges score every labelled reply again, with the
current prompts, and see where they disagree with the labels:

    uv run python -m evals.calibration compare

Each comparison is saved to calibration/comparisons/.
"""

import copy
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import openai
import yaml
from dotenv import load_dotenv

from agent.config import load_clinic_config, location_names
from backend.seed import DB_PATH
from evals.availability import (
    check_availability_turn,
    find_open_slots,
    is_in_request,
    resolve_request,
)
from evals.checks import CheckResult
from evals.judges import (
    JUDGE_MODEL,
    JUDGE_PROMPT_NAMES,
    JUDGE_REASONING_EFFORT,
    ask_judge,
    judge_inputs_for,
    load_judge_prompts,
)
from evals.run import load_cases

CALIBRATION_DIR = Path(__file__).parent / "calibration"
SYNTHETIC_PATH = CALIBRATION_DIR / "synthetic.yaml"
LABELS_PATH = CALIBRATION_DIR / "labels.json"
COMPARISONS_DIR = CALIBRATION_DIR / "comparisons"

USAGE = (
    "Usage:\n"
    "  uv run python -m evals.calibration sheet evals/results/<run>.json\n"
    "  uv run python -m evals.calibration compare"
)

# A judge is trusted once it agrees with at least this share of the labels.
AGREEMENT_TARGET = 0.85
# For these judges a missed failure lets a wrong language or an invented fact
# through as a pass, so they must catch every failure that was labelled.
NO_MISSED_FAILURES = ["language-consistency", "grounded"]

# Enough open slots for the labeller to check an offered time against, without
# a wall of them. The count says how many there were in all.
MAX_OPEN_SLOTS_SHOWN = 12

# What each label means, shown at the top of the sheet. The same questions
# the judges are asked in judge_prompts.yaml, in short.
SCORE_QUESTIONS = {
    "language-consistency": (
        "Is the reply entirely in the required language? Names and addresses don't count."
    ),
    "grounded": (
        "Is every fact in the reply supported by the conversation, the tool results"
        " or the dates?"
    ),
    "task-success": (
        "Does the reply do what the expected behaviour describes, with the right"
        " facts? Answering the question with a wrong answer is a fail."
    ),
    "phone-style": (
        "1-3 short sentences, at most one question, no lists or ids, times said the way"
        " people speak, and 2-3 times when offering?"
    ),
    "availability-correct": (
        "Are the offered times real open slots that match the request? If none match,"
        " does the reply say so?"
    ),
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def load_synthetic(path: Path = SYNTHETIC_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as synthetic_file:
        return yaml.safe_load(synthetic_file)


def judged_scores_for_turn(case_record: dict[str, Any], turn_number: int) -> list[str]:
    """The judge scores the run gave this turn, so the sheet asks for the same ones."""
    # Read from the run, not worked out again here: a turn gets exactly the
    # labels the judges scored. Code checks have no per-turn verdicts.
    score_names = []
    for score_name, result in case_record["scores"].items():
        for verdict in result.get("turns", []):
            if verdict["turn"] == turn_number:
                score_names.append(score_name)
    return score_names


def conversation_lines(
    greeting: str, turn_results: list[dict[str, Any]], index: int
) -> list[str]:
    """The call up to the caller's message, one line per message for easy reading."""
    lines = [f"Agent: {greeting}"]
    for earlier_turn in turn_results[:index]:
        lines.append(f"Caller: {earlier_turn['caller']}")
        lines.append(f"Agent: {earlier_turn['reply']}")
    lines.append(f"Caller: {turn_results[index]['caller']}")
    return lines


def availability_context(
    availability: dict[str, Any], today: date, now: datetime, db_path: Path
) -> dict[str, Any]:
    """The request, and the open slots that match it, for checking offered times."""
    request = resolve_request(availability, today)
    matching = []
    for slot in find_open_slots(db_path, request["appointment_type"], now):
        if is_in_request(slot, request):
            matching.append(slot[1])
    matching.sort()
    return {
        "request": request,
        "open_matching_count": len(matching),
        "open_matching_slots": matching[:MAX_OPEN_SLOTS_SHOWN],
    }


def label_item(
    item_id: str,
    case: dict[str, Any],
    case_record: dict[str, Any],
    turn_results: list[dict[str, Any]],
    index: int,
    greeting: str,
    run_record: dict[str, Any],
    db_path: Path,
) -> dict[str, Any]:
    """One reply to label, with everything a person needs to judge it."""
    turn_spec = case["turns"][index]
    tool_calls_so_far = []
    for turn_result in turn_results[: index + 1]:
        tool_calls_so_far.extend(turn_result["tool_calls"])

    if turn_spec.get("unclear"):
        required_language = "the fixed bilingual line"
    else:
        required_language = turn_spec.get("reply_language", case["language"])

    item = {
        "item_id": item_id,
        "required_language": required_language,
        "conversation": conversation_lines(greeting, turn_results, index),
        # Kept as JSON objects, not text, so the sheet shows them indented
        # instead of as one long line full of escaped quotes.
        "tool_results_so_far": tool_calls_so_far,
        "expected_behaviour": turn_spec["expected_behaviour"],
        "reply": turn_results[index]["reply"],
    }
    if "availability" in turn_spec:
        item["availability"] = availability_context(
            turn_spec["availability"],
            date.fromisoformat(run_record["today"]),
            datetime.fromisoformat(run_record["started_at"]),
            db_path,
        )

    labels = {}
    for score_name in judged_scores_for_turn(case_record, index + 1):
        labels[score_name] = None
    item["labels"] = labels
    item["notes"] = ""
    return item


def build_label_sheet(
    run_record: dict[str, Any],
    run_path: Path,
    cases: list[dict[str, Any]],
    synthetic: list[dict[str, Any]],
    greeting: str,
    db_path: Path,
) -> dict[str, Any]:
    cases_by_id = {}
    for case in cases:
        cases_by_id[case["id"]] = case

    items = []
    for case_record in run_record["cases"]:
        case = cases_by_id[case_record["id"]]
        turn_results = case_record["turns"]
        for index in range(len(turn_results)):
            turn_number = index + 1
            items.append(
                label_item(
                    f"{case['id']}/turn-{turn_number}",
                    case, case_record, turn_results, index, greeting, run_record, db_path,
                )
            )
            # Each hand-written reply goes right after the real one it
            # replaces, so the two can be compared.
            alternative_number = 0
            for entry in synthetic:
                if entry["case_id"] != case["id"] or entry["turn"] != turn_number:
                    continue
                alternative_number += 1
                # deepcopy: the replacement must not change the real run's turns.
                changed_turns = copy.deepcopy(turn_results)
                changed_turns[index]["reply"] = entry["reply"]
                items.append(
                    label_item(
                        f"{case['id']}/turn-{turn_number}/alt-{alternative_number}",
                        case, case_record, changed_turns, index, greeting, run_record, db_path,
                    )
                )

    return {
        "run_file": run_path.as_posix(),
        "how_to_label": (
            'Replace every null under "labels" with "pass" or "fail", judging the'
            ' reply only. Add anything worth knowing to "notes".'
        ),
        "questions": SCORE_QUESTIONS,
        "items": items,
    }


def write_label_sheet(sheet: dict[str, Any], path: Path = LABELS_PATH) -> None:
    # Labels take a person time, so never overwrite a sheet that may hold them.
    if path.exists():
        raise FileExistsError(f"{path} already exists. Move it away to build a new sheet.")
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as sheet_file:
        json.dump(sheet, sheet_file, ensure_ascii=False, indent=2)


# --- Comparing the judges with the labels ---


def parse_item_id(item_id: str) -> tuple[str, int]:
    """The case id and turn number in an item id such as "en-case/turn-2/alt-1"."""
    parts = item_id.split("/")
    case_id = parts[0]
    turn_number = int(parts[1].removeprefix("turn-"))
    return case_id, turn_number


def turns_as_labelled(
    case_record: dict[str, Any], item: dict[str, Any], index: int
) -> list[dict[str, Any]]:
    """The run's turns, with the reply that was labelled at `index`."""
    # For a hand-written alternative, the labelled reply isn't the run's.
    # Taking it from the sheet means the judge sees exactly what was labelled.
    turns = copy.deepcopy(case_record["turns"])
    turns[index]["reply"] = item["reply"]
    return turns


def judge_one_label(
    judge_client: openai.OpenAI,
    prompts: dict[str, str],
    case: dict[str, Any],
    turns: list[dict[str, Any]],
    index: int,
    score_name: str,
    greeting: str,
    names_of_locations: list[str],
    today: date,
    now: datetime,
    db_path: Path,
) -> CheckResult:
    """Score one labelled reply the way the eval run scores it."""
    if score_name == "availability-correct":
        return check_availability_turn(
            judge_client,
            prompts["extract_offered_times"],
            turns[index],
            case["turns"][index]["availability"],
            names_of_locations,
            today,
            db_path,
            now,
        )
    judge_inputs = judge_inputs_for(score_name, case, turns, greeting, today)
    instructions = prompts[JUDGE_PROMPT_NAMES[score_name]]
    return ask_judge(judge_client, instructions, judge_inputs[index])


def compare_sheet(
    judge_client: openai.OpenAI,
    prompts: dict[str, str],
    sheet: dict[str, Any],
    run_record: dict[str, Any],
    cases: list[dict[str, Any]],
    greeting: str,
    names_of_locations: list[str],
    db_path: Path,
) -> list[dict[str, Any]]:
    """Judge every labelled reply again, and pair each verdict with its label."""
    cases_by_id = {}
    for case in cases:
        cases_by_id[case["id"]] = case
    records_by_id = {}
    for case_record in run_record["cases"]:
        records_by_id[case_record["id"]] = case_record
    # The run's dates, not today's, so "next week" and the open slots are the
    # ones the agent and the labeller saw.
    today = date.fromisoformat(run_record["today"])
    now = datetime.fromisoformat(run_record["started_at"])

    comparisons = []
    for item in sheet["items"]:
        case_id, turn_number = parse_item_id(item["item_id"])
        index = turn_number - 1
        case = cases_by_id[case_id]
        turns = turns_as_labelled(records_by_id[case_id], item, index)
        for score_name, label in item["labels"].items():
            verdict = judge_one_label(
                judge_client, prompts, case, turns, index, score_name,
                greeting, names_of_locations, today, now, db_path,
            )
            comparison = {
                "item_id": item["item_id"],
                "score": score_name,
                "label": label,
                "judge": "pass" if verdict["passed"] else "fail",
                "judge_reason": verdict["reason"],
                "notes": item["notes"],
            }
            if "extracted" in verdict:
                comparison["extracted"] = verdict["extracted"]
            comparisons.append(comparison)
        # A full comparison takes minutes; this shows it's still going.
        print(f"  judged {item['item_id']}")
    return comparisons


def agreement_by_score(comparisons: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per score: labels, agreements, and the two kinds of disagreement."""
    counts: dict[str, dict[str, int]] = {}
    for comparison in comparisons:
        score_name = comparison["score"]
        if score_name not in counts:
            counts[score_name] = {
                "labels": 0, "agreed": 0, "missed_failures": 0, "false_alarms": 0
            }
        score_counts = counts[score_name]
        score_counts["labels"] += 1
        if comparison["judge"] == comparison["label"]:
            score_counts["agreed"] += 1
        elif comparison["label"] == "fail":
            # Labelled fail, judged pass: a bad reply would get through.
            score_counts["missed_failures"] += 1
        else:
            # Labelled pass, judged fail: a good reply flagged for nothing.
            score_counts["false_alarms"] += 1
    return counts


def meets_target(score_name: str, score_counts: dict[str, int]) -> bool:
    if score_counts["agreed"] / score_counts["labels"] < AGREEMENT_TARGET:
        return False
    if score_name in NO_MISSED_FAILURES and score_counts["missed_failures"] > 0:
        return False
    return True


def print_report(
    comparisons: list[dict[str, Any]], agreement: dict[str, dict[str, int]]
) -> None:
    print(f"\n=== Agreement with the labels (target {AGREEMENT_TARGET:.0%})")
    for score_name, score_counts in agreement.items():
        share = score_counts["agreed"] / score_counts["labels"]
        status = "OK  " if meets_target(score_name, score_counts) else "MISS"
        print(
            f"  {status} {score_name}: {score_counts['agreed']}/{score_counts['labels']}"
            f" ({share:.0%}), missed failures {score_counts['missed_failures']},"
            f" false alarms {score_counts['false_alarms']}"
        )

    print("\n=== Disagreements")
    for comparison in comparisons:
        if comparison["judge"] == comparison["label"]:
            continue
        print(
            f"\n  {comparison['item_id']}  {comparison['score']}:"
            f" labelled {comparison['label']}, judged {comparison['judge']}"
        )
        if comparison["notes"]:
            print(f"    Note:  {comparison['notes']}")
        print(f"    Judge: {comparison['judge_reason']}")


def save_comparison(record: dict[str, Any], comparisons_dir: Path = COMPARISONS_DIR) -> Path:
    comparisons_dir.mkdir(exist_ok=True)
    path = comparisons_dir / f"{record['comparison_id']}.json"
    with path.open("w", encoding="utf-8") as comparison_file:
        json.dump(record, comparison_file, ensure_ascii=False, indent=2)
    return path


# --- Commands ---


def make_sheet(run_path: Path) -> None:
    sheet = build_label_sheet(
        load_json(run_path),
        run_path,
        load_cases(),
        load_synthetic(),
        load_clinic_config()["greeting"],
        DB_PATH,
    )
    write_label_sheet(sheet)
    print(f"Wrote {len(sheet['items'])} replies to label to {LABELS_PATH}")


def compare() -> None:
    load_dotenv()  # the OpenAI key, for the judges
    sheet = load_json(LABELS_PATH)
    run_record = load_json(Path(sheet["run_file"]))
    config = load_clinic_config()
    started_at = datetime.now()

    comparisons = compare_sheet(
        openai.OpenAI(),
        load_judge_prompts(),
        sheet,
        run_record,
        load_cases(),
        config["greeting"],
        location_names(config),
        DB_PATH,
    )
    agreement = agreement_by_score(comparisons)
    print_report(comparisons, agreement)

    # Saved so each prompt change can be compared with the one before it.
    record = {
        "comparison_id": started_at.strftime("compare-%Y%m%d-%H%M%S"),
        "compared_at": started_at.isoformat(timespec="seconds"),
        "run_file": sheet["run_file"],
        "judge_model": JUDGE_MODEL,
        "judge_reasoning_effort": JUDGE_REASONING_EFFORT,
        "agreement": agreement,
        "comparisons": comparisons,
    }
    print(f"\nSaved to {save_comparison(record)}")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "sheet":
        make_sheet(Path(sys.argv[2]))
        return
    if len(sys.argv) == 2 and sys.argv[1] == "compare":
        compare()
        return
    sys.exit(USAGE)


if __name__ == "__main__":
    main()

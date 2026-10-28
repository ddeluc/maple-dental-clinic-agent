import json
from datetime import date

from agent.config import LANGUAGES, load_clinic_config, location_names
from agent.tools import APPOINTMENT_TYPES, TOOL_FUNCTIONS
from evals import run
from evals.availability import resolve_request
from evals.checks import check_language_detected, check_tool_use
from evals.run import load_cases, run_case, save_run, score_case, tool_calls_from


def call_item(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


def output_item(result: dict, call_id: str) -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": json.dumps(result)}


def turn_result(language: str | None, tool_calls: list[dict] | None = None) -> dict:
    return {"caller": "...", "reply": "...", "language": language, "tool_calls": tool_calls or []}


# --- cases.yaml ---


def test_every_case_is_well_formed():
    # A typo in cases.yaml should fail here, not halfway through a paid run.
    config = load_clinic_config()
    case_ids = set()
    for case in load_cases():
        assert case["id"] not in case_ids, f"duplicate id {case['id']}"
        case_ids.add(case["id"])
        assert case["description"]
        assert case["language"] in LANGUAGES
        assert case["turns"]
        for turn in case["turns"]:
            assert turn["caller"]
            assert turn["expected_behaviour"]
            if "reply_language" in turn:
                assert turn["reply_language"] in LANGUAGES
            for expected_tool in turn.get("expected_tools", []):
                assert expected_tool["name"] in TOOL_FUNCTIONS
            if "availability" in turn:
                availability = turn["availability"]
                assert availability["location"] in location_names(config)
                assert availability["appointment_type"] in APPOINTMENT_TYPES
                # Raises on a day word the runner can't turn into a date.
                resolve_request(availability, date(2026, 10, 27))


# --- tool_calls_from ---


def test_tool_calls_from_pairs_each_call_with_its_result():
    new_items = [
        call_item("get_clinic_info", {"location": "Verdun", "topic": "hours"}, "call_a"),
        call_item("get_clinic_info", {"location": "Plateau", "topic": "hours"}, "call_b"),
        # Results matched by call_id, not by position.
        output_item({"ok": True, "details": "Plateau hours"}, "call_b"),
        output_item({"ok": True, "details": "Verdun hours"}, "call_a"),
        {"role": "assistant", "content": "Both are open."},
    ]

    tool_calls = tool_calls_from(new_items)

    assert tool_calls == [
        {
            "name": "get_clinic_info",
            "arguments": {"location": "Verdun", "topic": "hours"},
            "result": {"ok": True, "details": "Verdun hours"},
        },
        {
            "name": "get_clinic_info",
            "arguments": {"location": "Plateau", "topic": "hours"},
            "result": {"ok": True, "details": "Plateau hours"},
        },
    ]


# --- check_language_detected ---


def test_language_detected_passes_when_every_turn_matches():
    case = {"language": "french", "turns": [{}, {}]}
    result = check_language_detected(case, [turn_result("french"), turn_result("french")])
    assert result["passed"]


def test_language_detected_fails_on_the_wrong_language():
    case = {"language": "english", "turns": [{}]}
    result = check_language_detected(case, [turn_result("french")])
    assert not result["passed"]
    assert "turn 1" in result["reason"]


def test_language_detected_expects_no_language_after_an_unclear_turn():
    case = {"language": "french", "turns": [{"unclear": True}, {}]}

    passing = check_language_detected(case, [turn_result(None), turn_result("french")])
    # Guessing a language on "Mm-hmm" is a failure even if the guess is right.
    guessing = check_language_detected(case, [turn_result("french"), turn_result("french")])

    assert passing["passed"]
    assert not guessing["passed"]


# --- check_tool_use ---


VERDUN_HOURS = {"name": "get_clinic_info", "arguments": {"location": "Verdun", "topic": "hours"}}


def test_tool_use_passes_when_the_expected_call_was_made():
    case = {"turns": [{"expected_tools": [VERDUN_HOURS]}]}
    result = check_tool_use(case, [turn_result("english", [VERDUN_HOURS])])
    assert result["passed"]


def test_tool_use_compares_only_the_listed_arguments():
    expected = {"name": "check_availability", "arguments": {"location": "Plateau"}}
    actual = {
        "name": "check_availability",
        "arguments": {"location": "Plateau", "first_day": "2026-11-02", "last_day": "2026-11-08"},
    }
    case = {"turns": [{"expected_tools": [expected]}]}
    assert check_tool_use(case, [turn_result("english", [actual])])["passed"]


def test_tool_use_fails_on_the_wrong_arguments_and_says_what_was_called():
    plateau_hours = {
        "name": "get_clinic_info",
        "arguments": {"location": "Plateau", "topic": "hours"},
    }
    case = {"turns": [{"expected_tools": [VERDUN_HOURS]}]}

    result = check_tool_use(case, [turn_result("english", [plateau_hours])])

    assert not result["passed"]
    assert "location='Plateau'" in result["reason"]


def test_tool_use_matches_a_null_location():
    # Insurers are the same at every location, so the case expects null.
    insurers = {"name": "get_clinic_info", "arguments": {"location": None, "topic": "insurers"}}
    case = {"turns": [{"expected_tools": [insurers]}]}
    assert check_tool_use(case, [turn_result("english", [insurers])])["passed"]


def test_tool_use_does_not_apply_when_no_turn_expects_tools():
    case = {"turns": [{}]}
    assert check_tool_use(case, [turn_result("english")]) is None


# --- run_case and score_case ---


def test_run_case_plays_each_turn_as_one_call(monkeypatch):
    config = load_clinic_config()
    case = {
        "id": "test-case",
        "language": "english",
        "turns": [{"caller": "Verdun hours?"}, {"caller": "Thanks!"}],
    }
    seen = []

    # Stands in for the agent: records what it was given and adds a tool
    # call on the first turn only, the way the real agent fills the history.
    def fake_answer(client, backend, config, tools, history, language, user_text, session_id):
        seen.append({"history_length": len(history), "language": language, "session": session_id})
        history.append({"role": "user", "content": user_text})
        if user_text == "Verdun hours?":
            history.append(call_item("get_clinic_info", VERDUN_HOURS["arguments"], "call_1"))
            history.append(output_item({"ok": True}, "call_1"))
        history.append({"role": "assistant", "content": f"reply to {user_text}"})
        return f"reply to {user_text}", "english"

    monkeypatch.setattr(run, "answer_and_trace", fake_answer)

    turn_results = run_case(None, None, config, [], case)

    # A fresh call: the greeting only, no language yet, one session for both turns.
    assert seen[0]["history_length"] == 1
    assert seen[0]["language"] is None
    assert seen[1]["language"] == "english"
    assert seen[0]["session"] == seen[1]["session"]
    assert seen[0]["session"].startswith("eval-test-case-")
    # Each turn gets only its own tool calls.
    assert turn_results[0]["tool_calls"][0]["arguments"] == VERDUN_HOURS["arguments"]
    assert turn_results[1]["tool_calls"] == []
    assert turn_results[1]["reply"] == "reply to Thanks!"


def test_each_run_of_a_case_is_its_own_session(monkeypatch):
    sessions = []

    def fake_answer(client, backend, config, tools, history, language, user_text, session_id):
        sessions.append(session_id)
        return "reply", "english"

    monkeypatch.setattr(run, "answer_and_trace", fake_answer)
    case = {"id": "test-case", "turns": [{"caller": "Hi"}]}
    config = {"greeting": "Bonjour, hi"}

    run_case(None, None, config, [], case)
    run_case(None, None, config, [], case)

    assert sessions[0] != sessions[1]


def test_save_run_writes_readable_json_named_after_the_run(tmp_path):
    run_record = {
        "run_id": "run-20261027-150000",
        "cases": [{"id": "fr-case", "turns": [turn_result("french")], "scores": {}}],
    }
    run_record["cases"][0]["turns"][0]["reply"] = "Oui, à Verdun."

    path = save_run(run_record, tmp_path / "results")

    assert path == tmp_path / "results" / "run-20261027-150000.json"
    text = path.read_text(encoding="utf-8")
    assert "Oui, à Verdun." in text  # accents kept, not à
    assert json.loads(text) == run_record


def test_score_case_leaves_out_checks_that_do_not_apply():
    case = {"language": "english", "turns": [{}]}
    scores = score_case(case, [turn_result("english")])
    assert list(scores) == ["language-detected"]

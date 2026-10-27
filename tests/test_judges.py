import json
from datetime import date
from types import SimpleNamespace

from evals.judges import (
    JUDGE_MODEL,
    ask_judge,
    extract_offered_times,
    grounded_inputs,
    judge_case,
    judge_each_turn,
    language_inputs,
    load_judge_prompts,
    task_inputs,
)

TODAY = date(2026, 10, 27)  # a Tuesday
GREETING = "Bonjour, hi!"


class FakeJudgeClient:
    """Stands in for the OpenAI client: plays back JSON answers in order."""

    def __init__(self, answers: list[dict]) -> None:
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.responses = self  # so client.responses.create() lands on create()

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(self.answers.pop(0)))


def passing(reason: str = "fine") -> dict:
    return {"reasoning": reason, "verdict": "pass"}


def failing(reason: str = "wrong") -> dict:
    return {"reasoning": reason, "verdict": "fail"}


def turn_result(caller: str, reply: str, tool_calls: list[dict] | None = None) -> dict:
    return {"caller": caller, "reply": reply, "language": "english", "tool_calls": tool_calls or []}


VERDUN_HOURS_CALL = {
    "name": "get_clinic_info",
    "arguments": {"location": "Verdun", "topic": "hours"},
    "result": {"ok": True, "details": "Monday to Friday 08:00-17:00"},
}


def test_judge_prompts_file_has_every_prompt():
    prompts = load_judge_prompts()
    for name in [
        "language_consistency",
        "grounded",
        "task_success",
        "phone_style",
        "extract_offered_times",
    ]:
        assert prompts[name].strip()


def test_ask_judge_uses_the_pinned_model_and_a_strict_schema():
    client = FakeJudgeClient([passing("all English")])

    verdict = ask_judge(client, "Check the language.", "Reply to check: Hi")

    call = client.calls[0]
    assert call["model"] == JUDGE_MODEL
    assert call["instructions"] == "Check the language."
    assert call["text"]["format"]["strict"] is True
    # Reasoning is listed before the verdict, so the model writes it first.
    assert list(call["text"]["format"]["schema"]["properties"]) == ["reasoning", "verdict"]
    assert verdict == {"passed": True, "reason": "all English"}


def test_ask_judge_turns_fail_into_not_passed():
    client = FakeJudgeClient([failing("has French")])
    assert ask_judge(client, "...", "...") == {"passed": False, "reason": "has French"}


def test_judge_each_turn_skips_turns_without_input_and_names_the_failing_turn():
    client = FakeJudgeClient([passing(), failing("invented a time")])

    result = judge_each_turn(client, "...", [None, "turn 2 input", "turn 3 input"])

    assert len(client.calls) == 2  # the skipped turn costs nothing
    assert not result["passed"]
    assert result["reason"] == "turn 3: invented a time"
    assert [verdict["turn"] for verdict in result["turns"]] == [2, 3]


def test_judge_each_turn_does_not_apply_when_every_turn_is_skipped():
    assert judge_each_turn(FakeJudgeClient([]), "...", [None, None]) is None


def test_language_inputs_follow_an_explicit_switch_and_skip_unclear_turns():
    case = {
        "language": "french",
        "turns": [{"unclear": True}, {}, {"reply_language": "english"}],
    }
    turns = [
        turn_result("Mm-hmm.", "Je peux vous aider... I can help..."),
        turn_result("Bonjour", "Bonjour!"),
        turn_result("English, please?", "Of course!"),
    ]

    judge_inputs = language_inputs(case, turns)

    assert judge_inputs[0] is None  # the bilingual line is meant to mix
    assert judge_inputs[1].startswith("Required language: french")
    assert judge_inputs[2].startswith("Required language: english")


def test_grounded_inputs_include_the_dates_and_earlier_tool_results():
    case = {"turns": [{}, {}]}
    turns = [
        turn_result("Verdun hours?", "Monday to Friday, 8 to 5.", [VERDUN_HOURS_CALL]),
        turn_result("And on Friday?", "Friday too, 8 to 5."),
    ]

    judge_inputs = grounded_inputs(case, turns, GREETING, TODAY)

    # The second reply reuses the first turn's result, so the judge must see it.
    assert "Monday to Friday 08:00-17:00" in judge_inputs[1]
    assert "Today is Tuesday, 2026-10-27." in judge_inputs[1]
    assert "Caller: And on Friday?" in judge_inputs[1]
    assert judge_inputs[1].endswith("Reply to check:\nFriday too, 8 to 5.")


def test_task_inputs_show_the_expected_behaviour_and_only_this_turns_tools():
    case = {"turns": [{"expected_behaviour": "Gives the hours."}, {"expected_behaviour": "Says goodbye."}]}
    turns = [
        turn_result("Verdun hours?", "8 to 5.", [VERDUN_HOURS_CALL]),
        turn_result("Thanks, bye!", "Goodbye!"),
    ]

    judge_inputs = task_inputs(case, turns, GREETING)

    assert "Expected behaviour:\nSays goodbye." in judge_inputs[1]
    assert "Tool results for this reply:\n(none)" in judge_inputs[1]
    # The conversation leads up to the caller's message, not past the reply.
    assert "Agent: 8 to 5.\nCaller: Thanks, bye!" in judge_inputs[1]


def test_judge_case_leaves_out_judges_that_skipped_every_turn():
    prompts = load_judge_prompts()
    case = {
        "language": "french",
        "turns": [{"unclear": True, "expected_behaviour": "The bilingual line."}],
    }
    turns = [turn_result("Mm-hmm.", "Je peux vous aider... I can help...")]
    client = FakeJudgeClient([passing()])

    scores = judge_case(client, prompts, case, turns, GREETING, TODAY)

    # Only task-success judges the fixed line; the other judges skip it.
    assert list(scores) == ["task-success"]
    assert len(client.calls) == 1


def test_extract_offered_times_allows_only_real_locations():
    answer = {
        "offered": [{"date": "2026-11-02", "start_time": "09:00", "location": "Plateau"}],
        "says_request_unavailable": False,
    }
    client = FakeJudgeClient([answer])

    extracted = extract_offered_times(
        client,
        "Read the times.",
        "Monday at 9 a.m.?",
        "A cleaning next week, please.",
        "Plateau",
        ["Plateau", "Verdun", "Saint-Laurent"],
        TODAY,
    )

    call = client.calls[0]
    schema = call["text"]["format"]["schema"]
    location_schema = schema["properties"]["offered"]["items"]["properties"]["location"]
    assert location_schema["enum"] == ["Plateau", "Verdun", "Saint-Laurent"]
    assert "The caller asked about the Plateau location." in call["input"]
    assert "Today is Tuesday, 2026-10-27." in call["input"]
    assert extracted == answer

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent.chat import (
    MAX_TOOL_ROUNDS,
    build_instructions,
    detect_language,
    prompt_for_language,
    run_chat,
    take_turn,
)
from agent.config import load_clinic_config, location_names


def text_response(text: str) -> SimpleNamespace:
    message = SimpleNamespace(type="message")
    return SimpleNamespace(output=[message], output_text=text)


def tool_call_response(name: str, arguments: dict, call_id: str = "call_1") -> SimpleNamespace:
    call = SimpleNamespace(
        type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id
    )
    return SimpleNamespace(output=[call], output_text="")


class FakeResponses:
    """Stands in for client.responses so tests never call the OpenAI API.

    It plays back the given responses in order, one per create() call.
    """

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        # Copy the history: take_turn keeps appending to the same list
        # afterwards, and we want to see what was sent at call time.
        # Language detection sends a plain string, which needs no copy.
        if isinstance(kwargs["input"], list):
            kwargs["input"] = list(kwargs["input"])
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = FakeResponses(responses)


def fake_backend(handler) -> httpx.Client:
    # MockTransport answers requests with `handler` instead of the network,
    # so these tests don't need the backend running.
    return httpx.Client(base_url="http://backend.test", transport=httpx.MockTransport(handler))


def clinic_info_handler(request: httpx.Request) -> httpx.Response:
    # The same shape the real /clinic-info endpoint returns.
    return httpx.Response(
        200,
        json={
            "topic": "parking",
            "location": "Verdun",
            "details": "Free lot behind the building.",
            "open_elsewhere": None,
        },
    )


def test_clinic_config_greets_in_french_first():
    config = load_clinic_config()
    assert config["greeting"].startswith("Bonjour, hi")
    assert config["system_prompt"].strip()


def test_clinic_config_lists_the_three_locations():
    config = load_clinic_config()
    assert location_names(config) == ["Plateau", "Verdun", "Saint-Laurent"]


def test_load_clinic_config_rejects_missing_language_rule(tmp_path: Path):
    config_path = tmp_path / "clinic.yaml"
    config_path.write_text(
        "system_prompt: You are a receptionist.\n"
        "greeting: Bonjour, hi!\n"
        "language_detection_prompt: Classify the language.\n"
        "unclear_language_reply: French or English?\n"
        "language_rules:\n"
        "  french: Reply only in French.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="english"):
        load_clinic_config(config_path)


def test_load_clinic_config_rejects_missing_greeting(tmp_path: Path):
    config_path = tmp_path / "clinic.yaml"
    config_path.write_text("system_prompt: You are a receptionist.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="greeting"):
        load_clinic_config(config_path)


def test_build_instructions_adds_today_and_next_week():
    instructions = build_instructions("System prompt", date(2026, 10, 23))
    assert instructions.startswith("System prompt")
    assert "Today is Friday, 2026-10-23. Next week is 2026-10-26 to 2026-11-01." in instructions


def test_build_instructions_lists_coming_days_with_weekdays():
    instructions = build_instructions("System prompt", date(2026, 10, 23))
    # Starts tomorrow and runs CALENDAR_DAYS ahead.
    assert "The coming days are: Saturday 2026-10-24, Sunday 2026-10-25," in instructions
    assert "Tuesday 2026-10-27" in instructions
    assert instructions.endswith("Friday 2026-11-06.")


def test_build_instructions_on_a_monday_next_week_starts_in_seven_days():
    instructions = build_instructions("System prompt", date(2026, 10, 26))
    assert "Next week is 2026-11-02 to 2026-11-08." in instructions


def test_take_turn_without_tools_sends_full_history_and_records_reply():
    client = FakeClient([text_response("Bien sûr, je peux vous aider.")])
    history = [{"role": "assistant", "content": "Bonjour, hi!"}]

    reply = take_turn(
        client, fake_backend(clinic_info_handler), "System prompt", [], history,
        "Je voudrais un rendez-vous.",
    )

    assert reply == "Bien sûr, je peux vous aider."
    sent = client.responses.calls[0]
    assert sent["instructions"].startswith("System prompt")
    assert sent["input"] == [
        {"role": "assistant", "content": "Bonjour, hi!"},
        {"role": "user", "content": "Je voudrais un rendez-vous."},
    ]
    assert history[-1] == {"role": "assistant", "content": "Bien sûr, je peux vous aider."}


def test_take_turn_runs_requested_tool_and_sends_result_back(capsys):
    client = FakeClient(
        [
            tool_call_response("get_clinic_info", {"location": "Verdun", "topic": "parking"}),
            text_response("There's a free lot behind the building."),
        ]
    )
    history = []

    reply = take_turn(
        client, fake_backend(clinic_info_handler), "System prompt", [], history,
        "Where do I park in Verdun?",
    )

    assert reply == "There's a free lot behind the building."
    # The second model call must include the call and its result.
    second_input = client.responses.calls[1]["input"]
    assert second_input[1]["type"] == "function_call"
    assert second_input[1]["name"] == "get_clinic_info"
    output_item = second_input[2]
    assert output_item["type"] == "function_call_output"
    assert output_item["call_id"] == "call_1"
    assert json.loads(output_item["output"])["details"] == "Free lot behind the building."
    # Both the call and the result are printed for debugging.
    printed = capsys.readouterr().out
    assert "[tool]   get_clinic_info" in printed
    assert "[result]" in printed


def test_take_turn_tool_error_goes_back_to_model_as_data():
    def backend_down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = FakeClient(
        [
            tool_call_response("get_clinic_info", {"location": "Verdun", "topic": "hours"}),
            text_response("Sorry, I can't check that right now."),
        ]
    )
    history = []

    reply = take_turn(
        client, fake_backend(backend_down), "System prompt", [], history, "Hours in Verdun?"
    )

    assert reply == "Sorry, I can't check that right now."
    result = json.loads(history[2]["output"])
    assert result["ok"] is False
    assert "can't be reached" in result["error"]


def test_take_turn_forces_text_reply_after_too_many_tool_rounds():
    responses = []
    for round_number in range(MAX_TOOL_ROUNDS):
        responses.append(
            tool_call_response(
                "get_clinic_info", {"location": "Verdun", "topic": "parking"},
                call_id=f"call_{round_number}",
            )
        )
    responses.append(text_response("Here's what I found."))
    client = FakeClient(responses)

    reply = take_turn(
        client, fake_backend(clinic_info_handler), "System prompt", [], [], "Parking?"
    )

    assert reply == "Here's what I found."
    assert len(client.responses.calls) == MAX_TOOL_ROUNDS + 1
    assert client.responses.calls[-1]["tool_choice"] == "none"


# --- Choosing the call's language ---


def language_response(language: str) -> SimpleNamespace:
    return text_response(json.dumps({"language": language}))


def test_detect_language_sends_only_the_callers_words():
    client = FakeClient([language_response("english")])

    language = detect_language(client, "Classify the language.", "Hi, I need a cleaning")

    assert language == "english"
    sent = client.responses.calls[0]
    assert sent["instructions"] == "Classify the language."
    assert sent["input"] == "Hi, I need a cleaning"
    schema = sent["text"]["format"]["schema"]
    assert schema["properties"]["language"]["enum"] == ["french", "english", "unclear"]


def test_detect_language_unclear_is_none():
    client = FakeClient([language_response("unclear")])
    assert detect_language(client, "Classify the language.", "ok") is None


def test_prompt_for_language_adds_the_rule_once_chosen():
    config = load_clinic_config()
    assert prompt_for_language(config, None) == config["system_prompt"]
    english_prompt = prompt_for_language(config, "english")
    assert english_prompt.startswith(config["system_prompt"])
    assert english_prompt.endswith(config["language_rules"]["english"])


def type_lines(monkeypatch, lines: list[str]) -> None:
    """Make input() return these lines, then behave like the end of piped input."""
    remaining = list(lines)

    def fake_input(prompt: str) -> str:
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


def test_run_chat_keeps_the_first_language_when_caller_switches(monkeypatch):
    config = load_clinic_config()
    type_lines(monkeypatch, ["Hi, I'd like a cleaning.", "Merci, au revoir!"])
    client = FakeClient(
        [
            language_response("english"),
            text_response("Sure! Which location would you prefer?"),
            text_response("Goodbye!"),
        ]
    )

    run_chat(client, fake_backend(clinic_info_handler), config)

    # One detection call, then one model call per caller message: the
    # French second message doesn't trigger a new detection.
    calls = client.responses.calls
    assert len(calls) == 3
    english_rule = config["language_rules"]["english"]
    assert english_rule in calls[1]["instructions"]
    assert english_rule in calls[2]["instructions"]


def test_run_chat_detects_again_until_the_language_is_clear(monkeypatch, capsys):
    config = load_clinic_config()
    type_lines(monkeypatch, ["Hola", "Bonjour, je voudrais un nettoyage."])
    client = FakeClient(
        [
            language_response("unclear"),
            language_response("french"),
            text_response("Bien sûr! Quelle clinique préférez-vous?"),
        ]
    )

    run_chat(client, fake_backend(clinic_info_handler), config)

    # The unclear message gets the fixed bilingual reply, with no model call.
    assert f"Agent: {config['unclear_language_reply']}" in capsys.readouterr().out
    calls = client.responses.calls
    assert len(calls) == 3
    assert calls[0]["input"] == "Hola"
    assert calls[1]["input"] == "Bonjour, je voudrais un nettoyage."
    # The model's call has the French rule and the earlier exchange.
    assert config["language_rules"]["french"] in calls[2]["instructions"]
    assert calls[2]["input"][1] == {"role": "user", "content": "Hola"}
    assert calls[2]["input"][2] == {
        "role": "assistant",
        "content": config["unclear_language_reply"],
    }

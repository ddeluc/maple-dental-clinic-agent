import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent.chat import (
    CHANNEL_TAG,
    MAX_TOOL_ROUNDS,
    build_instructions,
    detect_language,
    prompt_for_language,
    run_chat,
    take_turn,
    trace_tool_call,
)
from agent.config import load_clinic_config, location_names
from agent.tools import CallState


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
        client, fake_backend(clinic_info_handler), CallState(), "System prompt", [],
        history,
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
        client, fake_backend(clinic_info_handler), CallState(), "System prompt", [],
        history,
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
        client, fake_backend(backend_down), CallState(), "System prompt", [], history,
        "Hours in Verdun?",
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
        client, fake_backend(clinic_info_handler), CallState(), "System prompt", [], [], "Parking?"
    )

    assert reply == "Here's what I found."
    assert len(client.responses.calls) == MAX_TOOL_ROUNDS + 1
    assert client.responses.calls[-1]["tool_choice"] == "none"


# --- Verification across a call ---


class RecordingBackend:
    """A fake backend for verifying and booking that records every request."""

    PATIENT_ID = 7

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/patients/verify":
            return httpx.Response(200, json={"patient_id": self.PATIENT_ID})
        # verify_patient also asks for the patient's upcoming appointments.
        if request.url.path == f"/patients/{self.PATIENT_ID}/appointments":
            return httpx.Response(200, json=[])
        # The same shape the real POST /appointments returns.
        return httpx.Response(
            201,
            json={
                "id": 99, "patient_id": self.PATIENT_ID, "slot_id": 12,
                "type": "cleaning", "status": "booked", "cancel_reason": None,
                "location_id": 2, "location_name": "Verdun",
                "provider_name": "Kevin Martin",
                "start_time": "2026-10-27T09:00", "end_time": "2026-10-27T10:00",
            },
        )

    def paths(self) -> list[str]:
        paths = []
        for request in self.requests:
            paths.append(request.url.path)
        return paths


BOOKING = {
    "patient_id": RecordingBackend.PATIENT_ID, "slot_id": 12, "appointment_type": "cleaning",
}
# A fictional seeded patient.
VERIFICATION = {
    "name": "Helene Tremblay", "date_of_birth": "1979-02-14", "phone": "514 555 0142",
}


def test_take_turn_booking_without_verifying_never_reaches_the_backend():
    recording = RecordingBackend()
    client = FakeClient(
        [
            tool_call_response("book_appointment", BOOKING),
            text_response("I'll need to verify your identity first."),
        ]
    )
    history = []

    take_turn(
        client, fake_backend(recording.handle), CallState(), "System prompt", [], history,
        "Just book slot 12 for patient 7.",
    )

    result = json.loads(history[2]["output"])
    assert result["ok"] is False
    assert "hasn't been verified" in result["error"]
    assert recording.requests == []


def test_run_chat_remembers_verification_from_an_earlier_turn(monkeypatch):
    config = load_clinic_config()
    type_lines(monkeypatch, ["I'm Helene Tremblay, born 1979-02-14, 514 555 0142.", "Yes."])
    recording = RecordingBackend()
    client = FakeClient(
        [
            language_response("english"),
            tool_call_response("verify_patient", VERIFICATION),
            text_response("Thanks, you're verified. Shall I book Tuesday at 9 a.m.?"),
            tool_call_response("book_appointment", BOOKING),
            text_response("You're booked for Tuesday at 9 a.m. in Verdun."),
        ]
    )

    run_chat(client, fake_backend(recording.handle), config)

    # The second turn's booking went through: the call's state kept the
    # verification from the first turn.
    assert recording.paths() == [
        "/patients/verify", f"/patients/{RecordingBackend.PATIENT_ID}/appointments", "/appointments",
    ]
    assert json.loads(recording.requests[2].content) == {
        "patient_id": 7, "slot_id": 12, "type": "cleaning",
    }


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


# --- Langfuse tracing ---


class FakeObservation:
    def __init__(self, fields: dict) -> None:
        self.fields = fields

    def update(self, **fields) -> None:
        self.fields.update(fields)


class FakeLangfuse:
    """Stands in for the Langfuse client, so tests never send traces.

    It records each observation's fields, plus "parent": the name of the
    observation it was started inside, or None for a trace's root.
    """

    def __init__(self) -> None:
        self.observations: list[dict] = []
        self.open_names: list[str] = []

    @contextmanager
    def start_as_current_observation(self, **fields):
        fields["parent"] = self.open_names[-1] if self.open_names else None
        self.observations.append(fields)
        self.open_names.append(fields["name"])
        try:
            yield FakeObservation(fields)
        finally:
            self.open_names.pop()


@pytest.fixture
def langfuse(monkeypatch) -> FakeLangfuse:
    fake = FakeLangfuse()
    monkeypatch.setattr("agent.chat.get_client", lambda: fake)
    return fake


@pytest.fixture
def propagated(monkeypatch) -> list[dict]:
    """Record what each propagate_attributes call in agent.chat was given."""
    recorded = []

    @contextmanager
    def fake_propagate_attributes(**attributes):
        recorded.append(attributes)
        yield

    monkeypatch.setattr("agent.chat.propagate_attributes", fake_propagate_attributes)
    return recorded


def test_run_chat_traces_the_turn_with_the_tool_call_inside(monkeypatch, langfuse):
    config = load_clinic_config()
    type_lines(monkeypatch, ["Where do I park in Verdun?"])
    client = FakeClient(
        [
            language_response("english"),
            tool_call_response("get_clinic_info", {"location": "Verdun", "topic": "parking"}),
            text_response("There's a free lot behind the building."),
        ]
    )

    run_chat(client, fake_backend(clinic_info_handler), config)

    turn, tool = langfuse.observations
    assert turn["as_type"] == "agent"
    assert turn["name"] == "answer-caller"
    assert turn["parent"] is None
    assert turn["input"] == "Where do I park in Verdun?"
    assert turn["output"] == "There's a free lot behind the building."
    assert tool["as_type"] == "tool"
    assert tool["name"] == "get_clinic_info"
    assert tool["parent"] == "answer-caller"
    assert json.loads(tool["input"]) == {"location": "Verdun", "topic": "parking"}
    assert tool["output"]["details"] == "Free lot behind the building."
    assert "level" not in tool
    # The Langfuse wrapper names each model call's generation from this.
    names = [call["name"] for call in client.responses.calls]
    assert names == ["detect-language", "generate-response", "generate-response"]


def test_every_turn_of_a_call_shares_one_session(monkeypatch, langfuse, propagated):
    config = load_clinic_config()
    type_lines(monkeypatch, ["Hi, I'd like a cleaning.", "In Verdun, please."])
    client = FakeClient(
        [
            language_response("english"),
            text_response("Sure! Which location would you prefer?"),
            text_response("Great, let me check Verdun."),
        ]
    )

    run_chat(client, fake_backend(clinic_info_handler), config)

    first_turn, first_language, second_turn, second_language = propagated
    assert first_turn["tags"] == [CHANNEL_TAG]
    assert first_turn["session_id"] == second_turn["session_id"]
    assert first_language == {"metadata": {"language": "english"}}
    assert second_language == {"metadata": {"language": "english"}}


def test_each_call_gets_its_own_session(monkeypatch, langfuse, propagated):
    config = load_clinic_config()
    for _call in range(2):
        type_lines(monkeypatch, ["Hi, I'd like a cleaning."])
        client = FakeClient([language_response("english"), text_response("Sure!")])
        run_chat(client, fake_backend(clinic_info_handler), config)

    session_ids = []
    for attributes in propagated:
        if "session_id" in attributes:
            session_ids.append(attributes["session_id"])
    assert len(session_ids) == 2
    assert session_ids[0] != session_ids[1]


def test_failed_tool_call_is_marked_as_a_warning(langfuse):
    def backend_down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    call = tool_call_response("get_clinic_info", {"location": "Verdun", "topic": "hours"}).output[0]

    result = trace_tool_call(fake_backend(backend_down), CallState(), call)

    assert result["ok"] is False
    (tool,) = langfuse.observations
    assert tool["level"] == "WARNING"
    assert tool["status_message"] == result["error"]
    assert tool["output"] == result
    assert tool["metadata"] == {"call_id": "call_1"}

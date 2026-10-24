from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.chat import take_turn
from agent.config import load_clinic_config


class FakeResponses:
    """Stands in for client.responses so tests never call the OpenAI API."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        # Copy the history: take_turn appends the reply to the same list
        # afterwards, and we want to see what was sent at call time.
        kwargs["input"] = list(kwargs["input"])
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.reply)


class FakeClient:
    def __init__(self, reply: str) -> None:
        self.responses = FakeResponses(reply)


def test_clinic_config_greets_in_french_first():
    config = load_clinic_config()
    assert config["greeting"].startswith("Bonjour, hi")
    assert config["system_prompt"].strip()


def test_load_clinic_config_rejects_missing_greeting(tmp_path: Path):
    config_path = tmp_path / "clinic.yaml"
    config_path.write_text("system_prompt: You are a receptionist.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="greeting"):
        load_clinic_config(config_path)


def test_take_turn_sends_full_history_and_records_reply():
    client = FakeClient(reply="Bien sûr, je peux vous aider.")
    history = [{"role": "assistant", "content": "Bonjour, hi!"}]

    reply = take_turn(client, "System prompt", history, "Je voudrais un rendez-vous.")

    assert reply == "Bien sûr, je peux vous aider."
    sent = client.responses.calls[0]
    assert sent["instructions"] == "System prompt"
    assert sent["input"] == [
        {"role": "assistant", "content": "Bonjour, hi!"},
        {"role": "user", "content": "Je voudrais un rendez-vous."},
    ]
    assert history[-1] == {"role": "assistant", "content": "Bien sûr, je peux vous aider."}

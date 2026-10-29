import httpx
import pytest
from livekit.agents import llm

from agent import voice
from agent.config import load_clinic_config
from agent.voice import MapleDentalAgent, latest_caller_text


@pytest.fixture
def fake_keys(monkeypatch):
    # Building the agent creates LiveKit and OpenAI clients, which refuse to
    # start without keys. Nothing here calls either service, so fake ones do.
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    # 32+ characters, or the token library warns that the secret is too short.
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret-that-is-long-enough-for-jwt")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def make_chat_ctx(messages: list[tuple[str, str]]) -> llm.ChatContext:
    chat_ctx = llm.ChatContext.empty()
    for role, text in messages:
        chat_ctx.add_message(role=role, content=text)
    return chat_ctx


def test_latest_caller_text_is_the_last_thing_the_caller_said():
    chat_ctx = make_chat_ctx(
        [
            ("assistant", "Bonjour, hi"),
            ("user", "Hi, I need a cleaning."),
            ("assistant", "Sure, which location?"),
            ("user", "  Verdun, please.  "),
        ]
    )

    assert latest_caller_text(chat_ctx) == "Verdun, please."


def test_latest_caller_text_is_none_before_the_caller_speaks():
    chat_ctx = make_chat_ctx([("assistant", "Bonjour, hi")])

    assert latest_caller_text(chat_ctx) is None


def test_latest_caller_text_is_none_for_a_blank_turn():
    # A blank last turn means nothing to answer. Falling back to an earlier
    # message would answer the caller's previous question a second time.
    chat_ctx = make_chat_ctx([("user", "Hi, I need a cleaning."), ("user", "   ")])

    assert latest_caller_text(chat_ctx) is None


def test_each_turn_goes_through_answer_and_trace_with_the_voice_tag(monkeypatch, fake_keys):
    config = load_clinic_config()
    calls = []

    def fake_answer_and_trace(
        client, backend, call_state, config, tools, history, language, user_text,
        session_id, channel_tag,
    ):
        calls.append(
            {
                "language": language,
                "user_text": user_text,
                "session_id": session_id,
                "channel_tag": channel_tag,
                "history_start": history[0],
                "call_state": call_state,
            }
        )
        return f"Reply to: {user_text}", "english"

    monkeypatch.setattr(voice, "answer_and_trace", fake_answer_and_trace)
    with httpx.Client() as backend:
        agent = MapleDentalAgent(config, backend, session_id="room-123")
        first_reply = agent.answer("Hi, I need a cleaning.")
        agent.answer("Verdun, please.")

    assert first_reply == "Reply to: Hi, I need a cleaning."
    first_turn, second_turn = calls
    assert first_turn["channel_tag"] == "voice"
    assert first_turn["session_id"] == "room-123"
    # The history starts with the greeting, as in the text chat.
    assert first_turn["history_start"] == {"role": "assistant", "content": config["greeting"]}
    # The first turn finds the language; later turns keep it.
    assert first_turn["language"] is None
    assert second_turn["language"] == "english"
    # Both turns share one CallState, so a verification lasts the whole call.
    assert first_turn["call_state"] is second_turn["call_state"]


def test_each_call_starts_unverified_with_no_language(fake_keys):
    config = load_clinic_config()
    with httpx.Client() as backend:
        first_call = MapleDentalAgent(config, backend, session_id="room-1")
        second_call = MapleDentalAgent(config, backend, session_id="room-2")

    assert first_call.language is None
    assert first_call.call_state is not second_call.call_state

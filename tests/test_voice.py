from types import SimpleNamespace

import httpx
import pytest
from livekit.agents import ConversationItemAddedEvent, llm

from agent import voice
from agent.config import load_clinic_config
from agent.voice import MapleDentalAgent, latest_caller_text, log_if_interrupted, reply_language


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


def test_reply_language_recognizes_french():
    reply = "Parfait, votre rendez-vous est confirmé pour mardi à 9 h. Je vous attends!"

    assert reply_language(reply) == "fr"


def test_reply_language_recognizes_english():
    reply = "Great, your appointment is booked for Tuesday at 9. Is there anything else?"

    assert reply_language(reply) == "en"


def test_reply_language_ignores_french_place_names_in_an_english_reply():
    reply = "The Plateau clinic is on rue Saint-Denis. Which day works for you?"

    assert reply_language(reply) == "en"


def test_reply_language_is_none_when_it_cant_tell():
    # Only a time and a name: no marker words in either language.
    assert reply_language("9 h 30, Verdun.") is None


class FakeTTS:
    """Records the languages the agent sets on the text to speech."""

    def __init__(self) -> None:
        self.languages: list[str] = []

    def update_options(self, language: str) -> None:
        self.languages.append(language)


@pytest.fixture
def agent_with_fake_tts(monkeypatch, fake_keys):
    # Agent.session only works during a live LiveKit call. Replacing it on
    # the class with a plain object lets the test read what the agent sends
    # to the text to speech; monkeypatch puts the real property back after.
    fake_tts = FakeTTS()
    monkeypatch.setattr(MapleDentalAgent, "session", SimpleNamespace(tts=fake_tts))
    with httpx.Client() as backend:
        agent = MapleDentalAgent(load_clinic_config(), backend, session_id="room-1")
        yield agent, fake_tts


def test_voice_follows_a_switch_to_french_in_an_english_call(agent_with_fake_tts):
    agent, fake_tts = agent_with_fake_tts
    # The call started in English. The caller then asked to speak French,
    # so the model replies in French while self.language stays "english".
    agent.language = "english"

    agent.match_voice_to_reply("Sure, which location works best for you?")
    agent.match_voice_to_reply("Bien sûr! Quelle clinique vous convient le mieux?")

    assert fake_tts.languages == ["en", "fr"]


def test_voice_stays_french_before_the_language_is_chosen(agent_with_fake_tts):
    agent, fake_tts = agent_with_fake_tts
    unclear_reply = load_clinic_config()["unclear_language_reply"]

    agent.match_voice_to_reply(unclear_reply)

    # Nothing is changed, so the voice keeps the greeting's French.
    assert fake_tts.languages == []


def test_voice_is_unchanged_when_the_reply_language_is_unclear(agent_with_fake_tts):
    agent, fake_tts = agent_with_fake_tts
    agent.language = "english"

    agent.match_voice_to_reply("9 h 30, Verdun.")

    assert fake_tts.languages == []


def make_item_added_event(role: str, text: str, interrupted: bool) -> ConversationItemAddedEvent:
    message = llm.ChatMessage(role=role, content=[text], interrupted=interrupted)
    return ConversationItemAddedEvent(item=message)


def test_an_interrupted_reply_is_logged_with_what_was_heard(capsys):
    log_if_interrupted(make_item_added_event("assistant", "We have openings at 9,", True))

    assert capsys.readouterr().out == '  [interrupted] heard: "We have openings at 9,"\n'


def test_finished_replies_and_caller_turns_are_not_logged(capsys):
    log_if_interrupted(make_item_added_event("assistant", "We have openings at 9.", False))
    # The caller's own turns are never "interrupted" in the agent's sense.
    log_if_interrupted(make_item_added_event("user", "Hi, I need a cleaning.", True))

    assert capsys.readouterr().out == ""

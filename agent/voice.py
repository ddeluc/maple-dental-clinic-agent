"""Voice calls with the Maple Dental agent, through LiveKit.

Run with: uv run python -m agent.voice dev
The backend must be running too: uv run uvicorn backend.main:app --reload
Then start a session from the LiveKit Cloud agent console, with the agent
name "maple-dental".

LiveKit handles the audio: speech to text, turn detection, interruptions
and text to speech. Each caller turn is answered by answer_and_trace, the
same function the text chat uses, so both modes share one set of prompts,
tools and safety checks.
"""

import asyncio
import re
import threading
from typing import Any

import httpx
from dotenv import load_dotenv
from langfuse import get_client

# The same Langfuse-wrapped client as the text chat, so voice turns are
# traced like text turns.
from langfuse.openai import OpenAI
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    ModelSettings,
    TurnHandlingOptions,
    cli,
    inference,
    llm,
    room_io,
)
from livekit.plugins import ai_coustics

from agent.chat import Message, answer_and_trace
from agent.config import load_clinic_config, location_names
from agent.tools import CallState, make_backend_client, tool_definitions

# At import, not in main(): LiveKit answers each call in a new process,
# which imports this module but never runs main(). This copies the LiveKit,
# OpenAI and Langfuse keys from .env into the environment for every call.
load_dotenv()

# The name the LiveKit console (and later the phone line) asks for. LiveKit
# only sends a call to a worker registered under the name it asks for.
AGENT_NAME = "maple-dental"
CHANNEL_TAG = "voice"

# All three run through LiveKit Inference, so the LiveKit keys are the only
# ones needed for audio. nova-3 in "multi" mode hears French and English in
# one stream without choosing the language in advance.
STT_MODEL = "deepgram/nova-3"
STT_LANGUAGE = "multi"
TTS_MODEL = "cartesia/sonic-3"
TTS_VOICE = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"  # Cartesia's "Jacqueline"

# LiveKit skips the reply to every caller turn when no LLM is set. This one
# is never called: MapleDentalAgent.llm_node replaces it with our own turn
# logic, which uses its own OpenAI client.
UNUSED_LLM_MODEL = "openai/gpt-4o-mini"

# The text to speech must be told the language, or it reads English words
# with French pronunciation (or the reverse). The greeting starts in French.
GREETING_TTS_LANGUAGE = "fr"

# Short, common words that appear in almost every sentence of one language
# and not the other, used to tell which language a reply is in. Words both
# languages share (such as "a", "on" or "son") are left out on purpose.
FRENCH_MARKER_WORDS = {
    "je", "vous", "nous", "le", "la", "les", "un", "une", "des", "du", "de",
    "est", "et", "pour", "avec", "votre", "vos", "pas", "oui", "merci", "au",
    "bonjour", "rendez", "quel", "quelle", "ai", "êtes", "à",
}
ENGLISH_MARKER_WORDS = {
    "i", "you", "we", "the", "an", "is", "are", "and", "for", "with", "your",
    "not", "yes", "thanks", "thank", "hello", "appointment", "which", "what",
    "to", "of", "it", "have", "can",
}
# Letters only, accents included, so "l'heure" splits into "l" and "heure"
# and "rendez-vous" into "rendez" and "vous".
WORD_PATTERN = re.compile(r"[a-zàâäçéèêëîïôöûùüÿœæ]+")


def reply_language(reply: str) -> str | None:
    """The text to speech language code of a reply: "fr", "en", or None if unclear."""
    french_count = 0
    english_count = 0
    for word in WORD_PATTERN.findall(reply.lower()):
        if word in FRENCH_MARKER_WORDS:
            french_count += 1
        elif word in ENGLISH_MARKER_WORDS:
            english_count += 1

    if french_count > english_count:
        return "fr"
    if english_count > french_count:
        return "en"
    return None


def log_if_interrupted(event: ConversationItemAddedEvent) -> None:
    """Print what the caller heard when they cut the agent off."""
    item = event.item
    if item.type != "message" or item.role != "assistant":
        return
    # LiveKit marks a reply as interrupted and keeps only the part that was
    # actually spoken, so this shows where the agent stopped talking.
    if item.interrupted:
        print(f'  [interrupted] heard: "{item.text_content}"')


def latest_caller_text(chat_ctx: llm.ChatContext) -> str | None:
    """What the caller said in the turn that just ended, or None if nothing."""
    # LiveKit adds the caller's finished turn to the end of its history
    # before asking for a reply. Only that text is needed: our own history
    # (in MapleDentalAgent) keeps the rest, including tool calls, which
    # LiveKit's history never sees.
    for item in reversed(chat_ctx.items):
        if item.type == "message" and item.role == "user":
            text = item.text_content
            if text and text.strip():
                return text.strip()
            return None
    return None


class MapleDentalAgent(Agent):
    """One call: LiveKit's audio around the text agent's turn logic."""

    def __init__(self, config: dict[str, Any], backend: httpx.Client, session_id: str) -> None:
        super().__init__(
            # Required by LiveKit, but not sent to any model: the real
            # prompt is built each turn by the text agent's code.
            instructions="Answer through llm_node.",
            llm=inference.LLM(model=UNUSED_LLM_MODEL),
        )
        self.config = config
        self.backend = backend
        self.session_id = session_id
        self.client = OpenAI()
        # Not "tools": LiveKit's Agent already has a read-only `tools`
        # property for its own tool format, which this agent doesn't use.
        self.tool_specs = tool_definitions(location_names(config))
        # The same per-call state as run_chat in agent/chat.py: a history
        # that starts with the greeting, no language until the caller's
        # first clear message, and a fresh record of verified patients.
        self.history: list[Message] = [{"role": "assistant", "content": config["greeting"]}]
        self.language: str | None = None
        self.call_state = CallState()
        # One turn at a time. If the caller interrupts, LiveKit cancels the
        # waiting reply, but the thread answering it can't be stopped and
        # keeps changing the history. The next turn waits for it to finish.
        self.turn_lock = threading.Lock()

    async def on_enter(self) -> None:
        # Spoken from config instead of generated, as in the text chat, so
        # "Bonjour, hi" is always the first thing the caller hears.
        self.session.say(self.config["greeting"])

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> str | None:
        """Replace LiveKit's LLM step with the text agent's answer_and_trace."""
        caller_text = latest_caller_text(chat_ctx)
        if caller_text is None:
            return None  # nothing to answer, so the agent stays quiet

        # answer_and_trace blocks while it waits for OpenAI and the backend.
        # Run on the event loop, it would freeze the audio for the whole
        # call; in a thread, LiveKit keeps listening while it works.
        reply = await asyncio.to_thread(self.answer, caller_text)
        self.match_voice_to_reply(reply)
        return reply

    def answer(self, caller_text: str) -> str:
        with self.turn_lock:
            reply, self.language = answer_and_trace(
                self.client,
                self.backend,
                self.call_state,
                self.config,
                self.tool_specs,
                self.history,
                self.language,
                caller_text,
                self.session_id,
                channel_tag=CHANNEL_TAG,
            )
        return reply

    def match_voice_to_reply(self, reply: str) -> None:
        # Read from the reply, not from self.language: self.language never
        # changes after the first message, but the reply switches language
        # when the caller asks to (the prompt handles that). The voice must
        # follow what is actually about to be said.
        #
        # Until the call's language is chosen, the reply is always the
        # bilingual "unclear language" one, which starts in French. Counting
        # words in it could pick English, so the voice stays French instead.
        if self.language is None:
            return
        language_code = reply_language(reply)
        # Unclear (too few marker words to tell): keep the current voice.
        if language_code is None:
            return
        self.session.tts.update_options(language=language_code)


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def answer_call(ctx: JobContext) -> None:
    config = load_clinic_config()
    backend = make_backend_client()

    async def end_call() -> None:
        backend.close()
        # Langfuse sends traces in background batches. Flushing sends the
        # last ones before LiveKit ends this call's process.
        get_client().flush()

    ctx.add_shutdown_callback(end_call)

    session = AgentSession(
        stt=inference.STT(model=STT_MODEL, language=STT_LANGUAGE),
        tts=inference.TTS(model=TTS_MODEL, voice=TTS_VOICE, language=GREETING_TTS_LANGUAGE),
        turn_handling=TurnHandlingOptions(
            # Decides when the caller has finished speaking, from the audio
            # and the words, so a pause mid-sentence isn't taken as the end.
            turn_detection=inference.TurnDetector(),
            # How long a pause must last before the turn can end. Slower
            # than this detector's defaults (0.3s and 2.5s): callers pause
            # while reading out a phone number or date of birth, and
            # cutting them off mid-number is worse than a short wait.
            endpointing={"min_delay": 0.5, "max_delay": 3.0},
            interruption={
                # LiveKit's model that tells a real interruption from
                # "mm-hmm" or "oui, oui", which shouldn't stop the agent.
                # Set explicitly because LiveKit only turns it on by
                # default in dev mode, not when self-hosted in production.
                "mode": "adaptive",
                # Talking this long over the agent stops it. Short enough
                # to feel responsive, long enough that a cough doesn't.
                "min_duration": 0.5,
                # If the caller goes quiet right after a noise that stopped
                # the agent, finish the sentence instead of leaving silence.
                "resume_false_interruption": True,
                "false_interruption_timeout": 2.0,
            },
            # Off on purpose. Preemptive generation starts a reply before
            # the caller has finished, and throws it away if they keep
            # talking. Our replies run tools, so a thrown-away reply could
            # still have booked or cancelled an appointment.
            preemptive_generation={"enabled": False},
        ),
    )
    session.on("conversation_item_added", log_if_interrupted)

    # The room name is unique per session, so it doubles as the Langfuse
    # session id and makes a trace easy to match with LiveKit's call logs.
    agent = MapleDentalAgent(config, backend, session_id=ctx.room.name)
    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Same background-noise removal as LiveKit's quickstart.
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)

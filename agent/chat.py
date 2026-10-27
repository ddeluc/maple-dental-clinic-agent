"""Terminal chat with the Maple Dental agent.

Run with: uv run python -m agent.chat
The backend must be running too: uv run uvicorn backend.main:app --reload
Type a message and press Enter. Type "quit" or "exit" (or press Ctrl+C) to end.
"""

import json
import uuid
from datetime import date, timedelta
from typing import Any

import httpx
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes

# Langfuse's drop-in replacement for the OpenAI client: it behaves the same,
# and also records every model call (input, output, model, tokens, cost) as
# a "generation" in Langfuse. Without Langfuse keys it just stays quiet.
from langfuse.openai import OpenAI

from agent.config import LANGUAGES, load_clinic_config, location_names
from agent.tools import make_backend_client, run_tool, tool_definitions

MODEL = "gpt-4o-mini"
EXIT_WORDS = {"quit", "exit"}

# Names Langfuse shows for each kind of step. Evaluators, dashboards and
# saved filters find observations by name, so keep these stable.
TURN_NAME = "answer-caller"
DETECT_LANGUAGE_NAME = "detect-language"
GENERATE_RESPONSE_NAME = "generate-response"
# Tags are fixed when a trace starts. This one will tell text-chat traces
# apart from phone calls once the voice agent exists.
CHANNEL_TAG = "text-chat"

# Each turn, the model may ask for tools a few times in a row (for example
# Verdun's hours, then Plateau's). The cap stops a confused model from
# looping forever and running up the API bill.
MAX_TOOL_ROUNDS = 5

# How many days ahead the instructions list with their weekdays: enough
# for "next Friday" said on a Saturday.
CALENDAR_DAYS = 14

# One item of the conversation: a message such as
# {"role": "user", "content": "Bonjour"}, a tool call the model made, or
# that call's output. Every value in all three is a string.
Message = dict[str, str]


def build_instructions(system_prompt: str, today: date) -> str:
    # The model doesn't know today's date, and without it can't turn
    # "next week" or "next Tuesday" into the dates check_availability needs.
    # Even with today's date it often gets the arithmetic wrong, so the
    # code works out next week and a short calendar, and the model only
    # has to look a day up.
    days_until_next_monday = 7 - today.weekday()  # weekday(): Monday is 0
    next_monday = today + timedelta(days=days_until_next_monday)
    next_sunday = next_monday + timedelta(days=6)

    coming_days = []
    for offset in range(1, CALENDAR_DAYS + 1):
        day = today + timedelta(days=offset)
        coming_days.append(f"{day.strftime('%A')} {day.isoformat()}")

    return (
        f"{system_prompt}\n"
        f"Today is {today.strftime('%A')}, {today.isoformat()}. "
        f"Next week is {next_monday.isoformat()} to {next_sunday.isoformat()}.\n"
        f"The coming days are: {', '.join(coming_days)}."
    )


def detect_language(client: OpenAI, detection_prompt: str, user_text: str) -> str | None:
    """Return "french" or "english" for the caller's message, or None if unclear."""
    # A separate, narrow call that sees only the caller's words. The main
    # model, judging the language itself on every turn, was fooled by the
    # French greeting, French place names and English tool results.
    response = client.responses.create(
        # `name` is read and removed by the Langfuse wrapper; OpenAI never sees it.
        name=DETECT_LANGUAGE_NAME,
        model=MODEL,
        instructions=detection_prompt,
        input=user_text,
        # Structured output: the API makes the reply JSON that matches this
        # schema, so it is always exactly one of the three values.
        text={
            "format": {
                "type": "json_schema",
                "name": "caller_language",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "language": {"type": "string", "enum": list(LANGUAGES) + ["unclear"]},
                    },
                    "required": ["language"],
                    "additionalProperties": False,
                },
            }
        },
    )
    language = json.loads(response.output_text)["language"]
    if language == "unclear":
        return None
    return language


def prompt_for_language(config: dict[str, Any], language: str | None) -> str:
    """The system prompt, plus the rule that fixes the call's language once chosen."""
    if language is None:
        return config["system_prompt"]
    return f"{config['system_prompt']}\n{config['language_rules'][language]}"


def ask_model(
    client: OpenAI,
    instructions: str,
    history: list[Message],
    tools: list[dict[str, Any]],
    tool_choice: str = "auto",
) -> Any:
    # The API keeps no memory between calls, so every call sends the full
    # history, including earlier tool calls and their results.
    return client.responses.create(
        name=GENERATE_RESPONSE_NAME,  # for Langfuse, like in detect_language
        model=MODEL,
        instructions=instructions,
        input=history,
        tools=tools,
        tool_choice=tool_choice,
    )


def find_tool_calls(response: Any) -> list[Any]:
    """Return the tool calls in a response; an empty list means a text reply."""
    tool_calls = []
    for item in response.output:
        if item.type == "function_call":
            tool_calls.append(item)
    return tool_calls


def run_tool_calls(
    backend: httpx.Client, tool_calls: list[Any], history: list[Message]
) -> None:
    """Run each tool call, print it, and add the calls and results to the history."""
    # The calls go into the history first, then their outputs, as in the
    # OpenAI docs. Each output carries the call_id of the call it answers,
    # which is how the model matches them up when it asked for several.
    for call in tool_calls:
        history.append(
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            }
        )

    for call in tool_calls:
        print(f"  [tool]   {call.name} {call.arguments}")
        result = trace_tool_call(backend, call)
        # ensure_ascii=False keeps "Côté" readable instead of "Côté".
        result_json = json.dumps(result, ensure_ascii=False)
        print(f"  [result] {result_json}")
        history.append(
            {"type": "function_call_output", "call_id": call.call_id, "output": result_json}
        )


def trace_tool_call(backend: httpx.Client, call: Any) -> dict[str, Any]:
    """Run one tool call inside a Langfuse "tool" observation."""
    # The observation nests under the turn, next to the generation that
    # asked for it, so the trace reads: generation, tool, generation.
    with get_client().start_as_current_observation(
        as_type="tool",
        name=call.name,
        input=call.arguments,
        metadata={"call_id": call.call_id},
    ) as observation:
        result = run_tool(backend, call.name, call.arguments)
        observation.update(output=result)
        # Tools return errors as data, so nothing raises here. Marking the
        # observation lets you filter Langfuse for turns where a tool failed.
        if not result["ok"]:
            observation.update(level="WARNING", status_message=result["error"])
    return result


def record_reply(history: list[Message], reply: str) -> str:
    history.append({"role": "assistant", "content": reply})
    return reply


def take_turn(
    client: OpenAI,
    backend: httpx.Client,
    system_prompt: str,
    tools: list[dict[str, Any]],
    history: list[Message],
    user_text: str,
) -> str:
    """Answer one caller message, running any tools the model asks for."""
    history.append({"role": "user", "content": user_text})
    instructions = build_instructions(system_prompt, date.today())

    # The model either replies with text, which ends the turn, or asks for
    # tools. Then we run them, add the results to the history, and ask
    # again so it can use what the tools returned.
    for _round in range(MAX_TOOL_ROUNDS):
        response = ask_model(client, instructions, history, tools)
        tool_calls = find_tool_calls(response)
        if not tool_calls:
            return record_reply(history, response.output_text)
        run_tool_calls(backend, tool_calls, history)

    # Still asking for tools after MAX_TOOL_ROUNDS: tool_choice="none" makes
    # the model reply with text, using what it has learned so far.
    response = ask_model(client, instructions, history, tools, tool_choice="none")
    return record_reply(history, response.output_text)


def answer_caller(
    client: OpenAI,
    backend: httpx.Client,
    config: dict,
    tools: list[dict[str, Any]],
    history: list[Message],
    language: str | None,
    user_text: str,
) -> tuple[str, str | None]:
    """Answer one caller message. Returns the reply and the call's language."""
    if language is None:
        language = detect_language(client, config["language_detection_prompt"], user_text)
        if language is not None:
            print(f"  [language] {language}")

    # Still unclear: answer with the fixed bilingual line from config
    # instead of the model, which tended to answer in only one language.
    # Recording both messages lets the model see them on later turns.
    if language is None:
        reply = config["unclear_language_reply"]
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        return reply, None

    system_prompt = prompt_for_language(config, language)
    # Tags the turn and everything inside it with the call's language, so
    # Langfuse can compare French and English calls.
    with propagate_attributes(metadata={"language": language}):
        reply = take_turn(client, backend, system_prompt, tools, history, user_text)
    return reply, language


def answer_and_trace(
    client: OpenAI,
    backend: httpx.Client,
    config: dict,
    tools: list[dict[str, Any]],
    history: list[Message],
    language: str | None,
    user_text: str,
    session_id: str,
) -> tuple[str, str | None]:
    """Answer one caller message as one Langfuse trace in the call's session."""
    # Shared by the chat loop and the evals, so eval runs are traced the
    # same way as real conversations.
    #
    # propagate_attributes wraps the turn's root observation so the
    # session id and tag reach it and every observation inside it.
    # The root's input and output are what the caller said and what the
    # agent replied: the trace's summary in Langfuse's trace list. Its
    # duration is the turn's response time.
    with propagate_attributes(session_id=session_id, tags=[CHANNEL_TAG]):
        with get_client().start_as_current_observation(
            as_type="agent", name=TURN_NAME, input=user_text
        ) as turn:
            reply, language = answer_caller(
                client, backend, config, tools, history, language, user_text
            )
            turn.update(output=reply)
    return reply, language


def run_chat(client: OpenAI, backend: httpx.Client, config: dict) -> None:
    # The greeting comes from config instead of the model, so "Bonjour, hi"
    # is guaranteed to be the first thing the caller hears. It goes into the
    # history so the model knows it has already greeted the caller.
    greeting = config["greeting"]
    history: list[Message] = [{"role": "assistant", "content": greeting}]
    tools = tool_definitions(location_names(config))
    # None until a caller message is clearly French or English. After that
    # it never changes: it's the language the call started in. If the
    # caller explicitly asks to switch, the language rule in the prompt
    # tells the model to follow the request.
    language = None
    # Each turn is its own Langfuse trace. They all share this session id,
    # so Langfuse's Sessions view can replay the whole call in order.
    session_id = str(uuid.uuid4())
    print(f"Agent: {greeting}")

    while True:
        try:
            user_text = input("You: ").strip()
        except EOFError:  # Ctrl+D / Ctrl+Z, or the end of piped input
            print()
            return

        if user_text.lower() in EXIT_WORDS:
            return
        if not user_text:  # an accidental Enter shouldn't cost an API call
            continue

        reply, language = answer_and_trace(
            client, backend, config, tools, history, language, user_text, session_id
        )
        print(f"Agent: {reply}")


def main() -> None:
    # Before anything uses Langfuse: the Langfuse client reads its keys
    # from the environment when it's first used, inside run_chat.
    load_dotenv()  # copies the OpenAI and Langfuse keys from .env into the environment
    config = load_clinic_config()
    client = OpenAI()
    try:
        # "with" closes the backend connection when the chat ends, even on Ctrl+C.
        with make_backend_client() as backend:
            try:
                run_chat(client, backend, config)
            except KeyboardInterrupt:  # Ctrl+C, even while waiting for the model
                print()
    finally:
        # Langfuse sends traces in background batches. Flushing sends the
        # last ones before the program exits, or the final turns would be lost.
        get_client().flush()


if __name__ == "__main__":
    main()

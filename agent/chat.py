"""Terminal chat with the Maple Dental agent.

Run with: uv run python -m agent.chat
Type a message and press Enter. Type "quit" or "exit" (or press Ctrl+C) to end.
"""

from dotenv import load_dotenv
from openai import OpenAI

from agent.config import load_clinic_config

MODEL = "gpt-4o-mini"
EXIT_WORDS = {"quit", "exit"}

# One turn of the conversation, e.g. {"role": "user", "content": "Bonjour"}.
Message = dict[str, str]


def take_turn(
    client: OpenAI, system_prompt: str, history: list[Message], user_text: str
) -> str:
    """Send the caller's message with the whole conversation so far, record the reply."""
    history.append({"role": "user", "content": user_text})
    # The API keeps no memory between calls, so every call sends the full
    # history. That is what lets the model follow a language switch.
    response = client.responses.create(
        model=MODEL,
        instructions=system_prompt,
        input=history,
    )
    reply = response.output_text
    history.append({"role": "assistant", "content": reply})
    return reply


def run_chat(client: OpenAI, config: dict) -> None:
    # The greeting comes from config instead of the model, so "Bonjour, hi"
    # is guaranteed to be the first thing the caller hears. It goes into the
    # history so the model knows it has already greeted the caller.
    greeting = config["greeting"]
    history: list[Message] = [{"role": "assistant", "content": greeting}]
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

        reply = take_turn(client, config["system_prompt"], history, user_text)
        print(f"Agent: {reply}")


def main() -> None:
    load_dotenv()  # copies OPENAI_API_KEY from .env into the environment
    config = load_clinic_config()
    client = OpenAI()
    try:
        run_chat(client, config)
    except KeyboardInterrupt:  # Ctrl+C, even while waiting for the model
        print()


if __name__ == "__main__":
    main()

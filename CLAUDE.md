# CLAUDE.md

## Project
Maple Dental is an AI phone agent that answers a Montreal dental clinic's scheduling line in French or English. It books, reschedules and cancels appointments, answers clinic questions, and transfers to a human when it should.

- Project definition: `docs/project-definition.md` (clinic, agent behavior, tools, metrics, stack). It is the source of truth for what to build.

## How I work (read this first)
- **I'm building this to improve my Python and to be able to explain every part of it in an interview.** You can write any of the code, but I must understand all of it:
  - **Explain every change.** After writing code, explain what it does, why you chose this approach over the obvious alternatives, and how it fits with the rest of the project. Call out any Python feature or library behavior I might not know.
  - **Write for clarity, not cleverness.** Prefer plain loops over dense comprehensions, explicit names over short ones, small functions with one job, and early returns over deep nesting. No one-liners that need a second read, no metaprogramming, and no abstractions until there's a second real use.
  - **Comment the why, not the what.** Add a short comment wherever the reason for the code isn't obvious (for example, why verification is checked in code instead of the prompt).
  - When I write code myself, review it like a senior engineer: point out bugs, unclear code, and missing tests, and explain why.
- **When I give you a step with "Done when" criteria, they define the work.**
  - Work only on that step. Don't build ahead for later steps, even if you can see them.
  - When you finish, go through each "Done when" item one by one and show how it's met: a test run, a command and its output, or the row in the database. Don't just say it's done.
  - If an item isn't met or can't be checked yet, say so plainly. A step is done only when every item is met.
  - When every "Done when" item is met, suggest a commit message and commit only after I approve it. One commit per completed step.

## Structure
```
agent/      LiveKit agent and text chat loop: prompt loading, tools, escalation
backend/    FastAPI fake practice management system + SQLite (seed.py, clinic.db)
config/     Clinic-specific settings: maple_dental.yaml
docs/       Project definition, implementation plan (read only through /step)
scripts/    Helper scripts (get_step.py powers the /step command)
tests/      pytest tests
```

## Stack
- Python 3.11+
- FastAPI
- SQLite
- OpenAI Python SDK (OpenAI model with tool calling)
- LiveKit Agents
- Deepgram (multilingual speech to text)
- Cartesia (text to speech)
- Twilio (SIP trunk to LiveKit)
- Langfuse
- Fly.io
- pytest

See the project definition for how each is used.

## Commands
This project uses **uv** for Python, the virtual environment and dependencies. Never use `pip`, `pip install`, `requirements.txt` or manual `venv` activation. Dependencies live in `pyproject.toml`, locked in `uv.lock` (commit both).

- Install / sync dependencies: `uv sync`
- Add a dependency: `uv add <package>` (dev-only: `uv add --dev <package>`)
- Seed the database: `uv run python backend/seed.py`
- Run the backend: `uv run uvicorn backend.main:app --reload` (docs at http://localhost:8000/docs)
- Text agent: `uv run python -m agent.chat`
- Voice agent (dev): `uv run python -m agent.voice dev`
- Tests: `uv run pytest`

Update this section when a command changes.

## Rules for the code
- **Clinic-specific content lives in `config/`, never in Python.** Prompts, clinic info, escalation rules and greetings are loaded from YAML.
- **One set of tool functions** is shared by the text agent and the voice agent. Don't copy tools between modes.
- **Safety rules are enforced in code, not only in the prompt.** Booking, rescheduling and cancelling must fail if the patient isn't verified in the current conversation. Emergency language always leads to `transfer_to_human(queue="urgent")`.
- **The agent replies in the caller's language** and greets in French first ("Bonjour, hi").
- **Fake data only.** No real patient information anywhere, including tests and logs.
- **Secrets stay in `.env`**, which is git-ignored. Never print, log or commit API keys.
- Use type hints, small functions, and clear names. Readability beats brevity everywhere. Prefer the standard library and the existing stack before adding a dependency, and ask me before adding one (with `uv add`).
- Every backend endpoint and every tool gets at least one pytest test.

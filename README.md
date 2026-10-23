# Maple Dental Clinic Agent

An AI phone agent that answers a Montreal dental clinic's scheduling line in French or English. It books, reschedules and cancels appointments, answers clinic questions, and transfers to a human when needed. All patient data is fake.

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. Run `uv sync` to create the virtual environment and install dependencies.
3. Create a `.env` file in the project root with `OPENAI_API_KEY=...` (it is git-ignored).
4. Check the API key works: `uv run python scripts/hello_openai.py`

# Maple Dental Clinic Agent

An AI phone agent that answers a Montreal dental clinic's scheduling line in French or English. It books, reschedules and cancels appointments, answers clinic questions, and transfers to a human when needed. All patient data is fake.

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. Run `uv sync` to create the virtual environment and install dependencies.
3. Create a `.env` file in the project root (it is git-ignored) with:
   ```
   OPENAI_API_KEY=...
   # Optional: tracing in Langfuse. Keys are under Settings > API Keys in your Langfuse project.
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
   LANGFUSE_TRACING_ENVIRONMENT=development
   ```
   Without the Langfuse keys the agent still runs: Langfuse logs a warning and sends no traces.
4. Check the API key works: `uv run python scripts/hello_openai.py`

import os

# pytest runs this file before importing any test module. Turning Langfuse
# off here means no test ever sends traces, even when the shell has Langfuse
# keys set. Tests that check tracing swap in a fake Langfuse client instead.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

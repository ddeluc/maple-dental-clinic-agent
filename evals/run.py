"""Run the eval cases through the real agent and score every case.

Run with: uv run python -m evals.run
The backend must be running on a seeded database, as for the chat:
uv run python -m backend.seed, then uv run uvicorn backend.main:app
"""

import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import OpenAI

from agent.chat import Message, answer_and_trace
from agent.config import load_clinic_config, location_names
from agent.tools import make_backend_client, tool_definitions
from evals.checks import CheckResult, check_language_detected, check_tool_use

CASES_PATH = Path(__file__).parent / "cases.yaml"

# Eval traces go to their own Langfuse environment, so they never mix with
# real conversations in dashboards and filters.
EVAL_ENVIRONMENT = "eval"


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as cases_file:
        return yaml.safe_load(cases_file)


def tool_calls_from(new_items: list[Message]) -> list[dict[str, Any]]:
    """Pair each tool call in one turn's history items with its result."""
    # The history stores arguments and results as JSON strings, the way the
    # API wants them. The checks and judges need them as dicts.
    results_by_call_id = {}
    for item in new_items:
        if item.get("type") == "function_call_output":
            results_by_call_id[item["call_id"]] = json.loads(item["output"])

    tool_calls = []
    for item in new_items:
        if item.get("type") == "function_call":
            tool_calls.append(
                {
                    "name": item["name"],
                    "arguments": json.loads(item["arguments"]),
                    "result": results_by_call_id.get(item["call_id"]),
                }
            )
    return tool_calls


def run_case(
    client: OpenAI,
    backend: httpx.Client,
    config: dict[str, Any],
    tools: list[dict[str, Any]],
    case: dict[str, Any],
) -> list[dict[str, Any]]:
    """Play one case's caller messages to the agent, as one fresh call."""
    # The same start as a real call: the greeting is already in the history.
    history: list[Message] = [{"role": "assistant", "content": config["greeting"]}]
    language = None
    # The case id makes the session easy to find in Langfuse; the random
    # part keeps each run of the same case a separate session.
    session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"

    turn_results = []
    for turn in case["turns"]:
        # Everything the agent adds to the history after this index belongs
        # to this turn, which is how the turn's tool calls are found.
        first_new_item = len(history)
        reply, language = answer_and_trace(
            client, backend, config, tools, history, language, turn["caller"], session_id
        )
        turn_results.append(
            {
                "caller": turn["caller"],
                "reply": reply,
                "language": language,
                "tool_calls": tool_calls_from(history[first_new_item:]),
            }
        )
    return turn_results


def score_case(
    case: dict[str, Any], turn_results: list[dict[str, Any]]
) -> dict[str, CheckResult]:
    """Run every check on one case. Checks that don't apply are left out."""
    scores = {"language-detected": check_language_detected(case, turn_results)}
    tool_use = check_tool_use(case, turn_results)
    if tool_use is not None:
        scores["correct-tool-use"] = tool_use
    return scores


def print_case(
    case: dict[str, Any], turn_results: list[dict[str, Any]], scores: dict[str, CheckResult]
) -> None:
    print(f"\n=== {case['id']}")
    for turn_number, turn_result in enumerate(turn_results, start=1):
        print(f"  {turn_number}. Caller: {turn_result['caller']}")
        print(f"     Agent:  {turn_result['reply']}")
    for score_name, result in scores.items():
        verdict = "PASS" if result["passed"] else "FAIL"
        print(f"  {verdict}  {score_name}: {result['reason']}")


def print_summary(all_scores: list[dict[str, CheckResult]]) -> None:
    """Print each score's pass count, over the cases it applied to."""
    passed_by_name: dict[str, int] = {}
    scored_by_name: dict[str, int] = {}
    for scores in all_scores:
        for score_name, result in scores.items():
            scored_by_name[score_name] = scored_by_name.get(score_name, 0) + 1
            if result["passed"]:
                passed_by_name[score_name] = passed_by_name.get(score_name, 0) + 1

    print("\n=== Summary")
    for score_name, scored in scored_by_name.items():
        passed = passed_by_name.get(score_name, 0)
        print(f"  {score_name}: {passed}/{scored} cases passed")


def main() -> None:
    load_dotenv()
    # Set after load_dotenv, so it wins over the environment in .env, and
    # before anything creates the Langfuse client, which reads it once.
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = EVAL_ENVIRONMENT
    config = load_clinic_config()
    client = OpenAI()
    tools = tool_definitions(location_names(config))
    cases = load_cases()

    all_scores = []
    try:
        with make_backend_client() as backend:
            for case in cases:
                turn_results = run_case(client, backend, config, tools, case)
                scores = score_case(case, turn_results)
                print_case(case, turn_results, scores)
                all_scores.append(scores)
        print_summary(all_scores)
    finally:
        # Send the last eval traces before the program exits.
        get_client().flush()


if __name__ == "__main__":
    main()

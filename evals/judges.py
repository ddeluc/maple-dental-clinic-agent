"""LLM judges: the eval scores code can't decide, such as "is this all French?"

Each judge answers one narrow question about one agent reply, using the
instructions in judge_prompts.yaml. A case passes a judge when every turn
the judge applies to passes.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from agent.chat import describe_dates
from evals.checks import CheckResult, combine_turn_verdicts

JUDGE_PROMPTS_PATH = Path(__file__).parent / "judge_prompts.yaml"

# A stronger model than the agent's, so the judge catches mistakes the agent
# makes. Pinned to a dated version: if "gpt-5.5" moved to a new model, the
# scores would change without anything in this repo changing.
JUDGE_MODEL = "gpt-5.5-2026-04-23"
# The judges' questions are narrow, so a little reasoning is enough.
# Calibration (part 4) can raise it if a judge needs more.
JUDGE_REASONING_EFFORT = "low"

# Reasoning comes before the verdict: the model writes its answer in order,
# so the verdict is decided after the reasoning, not justified after the fact.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
    },
    "required": ["reasoning", "verdict"],
    "additionalProperties": False,
}


def load_judge_prompts(path: Path = JUDGE_PROMPTS_PATH) -> dict[str, str]:
    with path.open(encoding="utf-8") as prompts_file:
        return yaml.safe_load(prompts_file)


def ask_for_json(
    client: OpenAI, instructions: str, judge_input: str, schema_name: str, schema: dict
) -> dict[str, Any]:
    """Ask the judge model a question whose answer must match `schema`."""
    response = client.responses.create(
        model=JUDGE_MODEL,
        reasoning={"effort": JUDGE_REASONING_EFFORT},
        instructions=instructions,
        input=judge_input,
        # The same structured output as the agent's language detection:
        # the answer is always JSON in exactly this shape.
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )
    return json.loads(response.output_text)


def ask_judge(client: OpenAI, instructions: str, judge_input: str) -> CheckResult:
    answer = ask_for_json(client, instructions, judge_input, "verdict", VERDICT_SCHEMA)
    return {"passed": answer["verdict"] == "pass", "reason": answer["reasoning"]}


def judge_each_turn(
    client: OpenAI, instructions: str, judge_inputs: list[str | None]
) -> CheckResult | None:
    """Judge every turn that has an input; None means the judge skips that turn."""
    turn_verdicts = []
    for index, judge_input in enumerate(judge_inputs):
        if judge_input is None:
            continue
        verdict = ask_judge(client, instructions, judge_input)
        verdict["turn"] = index + 1
        turn_verdicts.append(verdict)
    return combine_turn_verdicts(turn_verdicts)


# --- What each judge is shown ---


def format_conversation(greeting: str, turn_results: list[dict[str, Any]]) -> str:
    lines = [f"Agent: {greeting}"]
    for turn_result in turn_results:
        lines.append(f"Caller: {turn_result['caller']}")
        lines.append(f"Agent: {turn_result['reply']}")
    return "\n".join(lines)


def format_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    if not tool_calls:
        return "(none)"
    lines = []
    for call in tool_calls:
        # ensure_ascii=False keeps "Côté" readable for the judge.
        arguments = json.dumps(call["arguments"], ensure_ascii=False)
        result = json.dumps(call["result"], ensure_ascii=False)
        lines.append(f"- {call['name']}({arguments}) returned {result}")
    return "\n".join(lines)


def conversation_before_reply(
    greeting: str, turn_results: list[dict[str, Any]], index: int
) -> str:
    """The call up to and including the caller's message that the reply answers."""
    earlier_turns = format_conversation(greeting, turn_results[:index])
    return f"{earlier_turns}\nCaller: {turn_results[index]['caller']}"


def language_inputs(case: dict[str, Any], turn_results: list[dict[str, Any]]) -> list[str | None]:
    judge_inputs = []
    for index, turn_result in enumerate(turn_results):
        turn_spec = case["turns"][index]
        # The bilingual line for an unclear message is meant to mix languages.
        if turn_spec.get("unclear"):
            judge_inputs.append(None)
            continue
        required_language = turn_spec.get("reply_language", case["language"])
        judge_inputs.append(
            f"Required language: {required_language}\n\n"
            f"Caller said:\n{turn_result['caller']}\n\n"
            f"Reply to check:\n{turn_result['reply']}"
        )
    return judge_inputs


def grounded_inputs(
    case: dict[str, Any], turn_results: list[dict[str, Any]], greeting: str, today: date
) -> list[str | None]:
    judge_inputs = []
    tool_calls_so_far = []
    for index, turn_result in enumerate(turn_results):
        # Facts from an earlier turn's tool results are still the agent's
        # to use, so the judge sees every result up to this reply.
        tool_calls_so_far.extend(turn_result["tool_calls"])
        if case["turns"][index].get("unclear"):
            judge_inputs.append(None)  # a fixed line from config, not the model's
            continue
        judge_inputs.append(
            f"Dates the agent was told:\n{describe_dates(today)}\n\n"
            f"Conversation:\n{conversation_before_reply(greeting, turn_results, index)}\n\n"
            f"Tool results so far:\n{format_tool_calls(tool_calls_so_far)}\n\n"
            f"Reply to check:\n{turn_result['reply']}"
        )
    return judge_inputs


def task_inputs(
    case: dict[str, Any], turn_results: list[dict[str, Any]], greeting: str
) -> list[str | None]:
    judge_inputs = []
    for index, turn_result in enumerate(turn_results):
        expected_behaviour = case["turns"][index]["expected_behaviour"]
        judge_inputs.append(
            f"Conversation:\n{conversation_before_reply(greeting, turn_results, index)}\n\n"
            f"Tool results for this reply:\n{format_tool_calls(turn_result['tool_calls'])}\n\n"
            f"Expected behaviour:\n{expected_behaviour}\n\n"
            f"Reply to check:\n{turn_result['reply']}"
        )
    return judge_inputs


def style_inputs(case: dict[str, Any], turn_results: list[dict[str, Any]]) -> list[str | None]:
    judge_inputs = []
    for index, turn_result in enumerate(turn_results):
        if case["turns"][index].get("unclear"):
            judge_inputs.append(None)  # a fixed line from config, not the model's
            continue
        judge_inputs.append(f"Reply to check:\n{turn_result['reply']}")
    return judge_inputs


# Each judge score and its instructions in judge_prompts.yaml.
JUDGE_PROMPT_NAMES = {
    "language-consistency": "language_consistency",
    "grounded": "grounded",
    "task-success": "task_success",
    "phone-style": "phone_style",
}


def judge_inputs_for(
    score_name: str,
    case: dict[str, Any],
    turn_results: list[dict[str, Any]],
    greeting: str,
    today: date,
) -> list[str | None]:
    """What one judge is shown for each turn of a case."""
    # One place that knows which inputs go with which judge, shared by the
    # eval run and by calibration, so the two always judge the same way.
    if score_name == "language-consistency":
        return language_inputs(case, turn_results)
    if score_name == "grounded":
        return grounded_inputs(case, turn_results, greeting, today)
    if score_name == "task-success":
        return task_inputs(case, turn_results, greeting)
    if score_name == "phone-style":
        return style_inputs(case, turn_results)
    raise ValueError(f"No judge for score '{score_name}'")


def judge_case(
    client: OpenAI,
    prompts: dict[str, str],
    case: dict[str, Any],
    turn_results: list[dict[str, Any]],
    greeting: str,
    today: date,
) -> dict[str, CheckResult]:
    """Run every judge on one case. Judges that skip every turn are left out."""
    scores = {}
    for score_name, prompt_name in JUDGE_PROMPT_NAMES.items():
        judge_inputs = judge_inputs_for(score_name, case, turn_results, greeting, today)
        result = judge_each_turn(client, prompts[prompt_name], judge_inputs)
        if result is not None:
            scores[score_name] = result
    return scores


# --- Reading offered times, for availability-correct ---


def offered_times_schema(location_names: list[str]) -> dict[str, Any]:
    # The locations are an enum, so a time can't come back at a location
    # that doesn't exist, the same idea as in the agent's tool definitions.
    return {
        "type": "object",
        "properties": {
            "offered": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "YYYY-MM-DD"},
                        "start_time": {"type": "string", "description": "HH:MM, 24-hour"},
                        "location": {"type": "string", "enum": location_names},
                    },
                    "required": ["date", "start_time", "location"],
                    "additionalProperties": False,
                },
            },
            "says_request_unavailable": {"type": "boolean"},
        },
        "required": ["offered", "says_request_unavailable"],
        "additionalProperties": False,
    }


def extract_offered_times(
    client: OpenAI,
    instructions: str,
    reply: str,
    caller_text: str,
    requested_location: str,
    location_names: list[str],
    today: date,
) -> dict[str, Any]:
    """Read the appointment times a reply offers, as dates and 24-hour times."""
    judge_input = (
        f"{describe_dates(today)}\n\n"
        f"The caller asked about the {requested_location} location. They said:\n"
        f"{caller_text}\n\n"
        f"Reply to read:\n{reply}"
    )
    return ask_for_json(
        client, instructions, judge_input, "offered_times", offered_times_schema(location_names)
    )

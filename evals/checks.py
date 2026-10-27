"""Code checks: the eval scores that code can decide exactly, without a judge.

Each check looks at one case's spec (from cases.yaml) and what the agent did
on each turn (from run.py), and returns {"passed": bool, "reason": str}, or
None when the case gives it nothing to check.
"""

from typing import Any

CheckResult = dict[str, Any]


def combine_turn_verdicts(turn_verdicts: list[CheckResult]) -> CheckResult | None:
    """One case result from per-turn verdicts: it passes if every turn passed.

    Each verdict has "passed", "reason" and "turn". None means no turn was
    scored, so the score doesn't apply to the case.
    """
    if not turn_verdicts:
        return None

    failures = []
    for verdict in turn_verdicts:
        if not verdict["passed"]:
            failures.append(f"turn {verdict['turn']}: {verdict['reason']}")
    # Each turn's verdict is kept, so a reviewer can see why every turn passed
    # or failed, not only the case's result.
    if failures:
        return {"passed": False, "reason": " | ".join(failures), "turns": turn_verdicts}
    return {"passed": True, "reason": "every turn passed", "turns": turn_verdicts}


def check_language_detected(
    case: dict[str, Any], turn_results: list[dict[str, Any]]
) -> CheckResult:
    """Did language detection settle on the case's language, at the right turn?"""
    problems = []
    for index, turn_result in enumerate(turn_results):
        turn_spec = case["turns"][index]
        # An unclear message must leave the language unset, so the caller
        # gets the bilingual line instead of a guess.
        if turn_spec.get("unclear"):
            expected_language = None
        else:
            expected_language = case["language"]
        if turn_result["language"] != expected_language:
            problems.append(
                f"turn {index + 1}: detected {turn_result['language']},"
                f" expected {expected_language}"
            )

    if problems:
        return {"passed": False, "reason": "; ".join(problems)}
    return {"passed": True, "reason": f"detected {case['language']}"}


def arguments_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    # Only the arguments the case lists are compared, so a case can leave
    # out the dates and accept any sensible date range.
    for name, value in expected.items():
        if name not in actual or actual[name] != value:
            return False
    return True


def was_called(expected_tool: dict[str, Any], tool_calls: list[dict[str, Any]]) -> bool:
    expected_arguments = expected_tool.get("arguments", {})
    for call in tool_calls:
        if call["name"] == expected_tool["name"] and arguments_match(
            expected_arguments, call["arguments"]
        ):
            return True
    return False


def describe_tool(tool: dict[str, Any]) -> str:
    """Write a tool call or expectation as get_clinic_info(location='Verdun', ...)."""
    arguments = []
    for name, value in tool.get("arguments", {}).items():
        arguments.append(f"{name}={value!r}")
    return f"{tool['name']}({', '.join(arguments)})"


def check_tool_use(
    case: dict[str, Any], turn_results: list[dict[str, Any]]
) -> CheckResult | None:
    """Did the agent call every tool the case expects, on the turn it expects?"""
    checked_any_turn = False
    problems = []
    for index, turn_result in enumerate(turn_results):
        expected_tools = case["turns"][index].get("expected_tools")
        if not expected_tools:
            continue
        checked_any_turn = True
        for expected_tool in expected_tools:
            if was_called(expected_tool, turn_result["tool_calls"]):
                continue
            actual_calls = []
            for call in turn_result["tool_calls"]:
                actual_calls.append(describe_tool(call))
            problems.append(
                f"turn {index + 1}: expected {describe_tool(expected_tool)},"
                f" called [{', '.join(actual_calls)}]"
            )

    if not checked_any_turn:
        return None
    if problems:
        return {"passed": False, "reason": "; ".join(problems)}
    return {"passed": True, "reason": "every expected tool was called"}

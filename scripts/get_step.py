"""Print one step or one phase from the implementation plan.

Usage:
    uv run python scripts/get_step.py 2.3   # prints only Step 2.3
    uv run python scripts/get_step.py 2     # prints Phase 2 and all of its steps

Used by the /step command so the coding agent only sees the part of the
plan we're working on, not the whole file.
"""

import sys
from pathlib import Path

# The plan lives at docs/implementation-plan.md, one folder up from scripts/.
PLAN_PATH = Path(__file__).parent.parent / "docs" / "implementation-plan.md"


def find_start_heading(identifier: str) -> tuple[str, int]:
    """Return the heading text to look for and its markdown level.

    "2.3" means a step, written in the plan as "### Step 2.3: ...".
    "2" means a phase, written in the plan as "## Phase 2: ...".
    """
    if "." in identifier:
        return f"### Step {identifier}:", 3
    return f"## Phase {identifier}:", 2


def heading_level(line: str) -> int:
    """Return how many '#' a markdown heading starts with, or 0 if it isn't a heading."""
    if not line.startswith("#"):
        return 0
    level = len(line) - len(line.lstrip("#"))
    # A real heading has a space after the #'s, like "## Phase 1".
    if not line[level:].startswith(" "):
        return 0
    return level


def extract_section(plan_lines: list[str], identifier: str) -> list[str]:
    """Return the lines of one step or phase, from its heading up to the next one."""
    start_heading, start_level = find_start_heading(identifier)

    section_lines = []
    inside_section = False

    for line in plan_lines:
        if not inside_section:
            if line.startswith(start_heading):
                inside_section = True
                section_lines.append(line)
            continue

        # Stop at the next heading at the same level or higher.
        # For a step that's the next "###" or "##"; for a phase it's the next "##",
        # so a phase keeps all of its "###" steps.
        level = heading_level(line)
        if level != 0 and level <= start_level:
            break

        section_lines.append(line)

    # Drop the "---" divider and blank lines that sit between sections.
    while section_lines and section_lines[-1].strip() in ("", "---"):
        section_lines.pop()

    return section_lines


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: get_step.py <step like 2.3, or phase like 2>")
        return 1

    identifier = sys.argv[1].strip()

    if not PLAN_PATH.exists():
        print(f"ERROR: plan not found at {PLAN_PATH}")
        return 1

    plan_lines = PLAN_PATH.read_text(encoding="utf-8").splitlines()
    section_lines = extract_section(plan_lines, identifier)

    if not section_lines:
        print(f"ERROR: no step or phase '{identifier}' in the implementation plan.")
        return 1

    print("\n".join(section_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

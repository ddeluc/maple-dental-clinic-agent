---
description: Work on one step or phase of the implementation plan
argument-hint: <step like 2.3, or phase like 2>
allowed-tools: Bash(uv run python scripts/get_step.py:*)
---

This is the part of the implementation plan we're working on now:

!`uv run python scripts/get_step.py $ARGUMENTS`

If the text above starts with "ERROR", stop and tell me. Don't guess what the step is, and don't open the plan file yourself.

Otherwise:
- Work only on what's described above. Don't build ahead for anything that isn't in it.
- Before writing code, give me a short plan of what you'll do, in a few bullets, and wait for my OK.
- Follow the "How I work" rules in CLAUDE.md: explain every change and write for clarity.
- When you're finished, go through each "Done when" item one by one and show how it's met (a test run, a command and its output, or a database row). If an item isn't met, say so plainly.

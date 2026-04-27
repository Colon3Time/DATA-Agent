from __future__ import annotations

from pathlib import Path
from typing import Callable


ANNA_OUTPUT_CONTRACT = """

---
## Anna Output Contract v3 (strict)

When the user request requires agent work, Anna MUST follow this contract:

1. Select or create a project before the first DISPATCH.
   - New task: emit CREATE_DIR for projects/<short_project_name>/input and projects/<short_project_name>/output.
   - Existing task: clearly reference the existing project name.
   - Never dispatch agents while project is unknown.

2. DISPATCH tags must contain one valid JSON object only:
   <DISPATCH>{"agent":"scout","task":"specific task with input/output context"}</DISPATCH>

3. Required DISPATCH fields:
   - agent: one of scout, dana, eddie, max, finn, mo, iris, vera, quinn, rex
   - task: concrete instruction with file paths, handoff context, or exact expected output

4. Optional DISPATCH fields:
   - discover: boolean true/false only
   - parallel_group: short string only, and only when agents can run safely in parallel

5. CRISP-DM guardrails:
   - If no dataset exists, dispatch scout first.
   - If dataset exists but no cleaning/EDA exists, prefer dana and/or eddie before modeling.
   - Do not dispatch mo unless Finn runs before Mo in the same plan, or valid Finn output already exists.
   - Do not dispatch quinn/rex final work before Mo has produced model output, unless user explicitly asks for reporting only.
   - If required context is missing, use READ_FILE or ASK_USER before dispatching.

6. Avoid low-value dispatch:
   - Do not dispatch agents for greetings, explanations, or normal chat.
   - Do not emit vague tasks such as "...", "analyze this", "ทำต่อ", or "วิเคราะห์" without context.
"""


ReadPipelineFn = Callable[[str], str]


def validate_dispatch_plan(
    dispatches: list[dict],
    *,
    active_project: Path | None,
    read_pipeline: ReadPipelineFn,
) -> list[str]:
    """Return issues that should force Anna to repair the plan before execution."""
    issues: list[str] = []
    if not dispatches:
        return issues

    if active_project is None:
        issues.append("No active project was selected or created before DISPATCH.")

    agents = [str(d.get("agent", "")).lower() for d in dispatches]
    tasks = [str(d.get("task", "")).strip() for d in dispatches]

    vague_tasks = {"...", "analyze this", "ทำต่อ", "วิเคราะห์", "analyze", "continue"}
    for idx, task in enumerate(tasks):
        if len(task) < 12 or task.lower() in vague_tasks:
            issues.append(f"Dispatch #{idx + 1} has a vague or placeholder task.")

    has_finn_before_mo = False
    existing_finn = bool(read_pipeline("finn"))
    for agent in agents:
        if agent == "finn":
            has_finn_before_mo = True
        if agent == "mo" and not (has_finn_before_mo or existing_finn):
            issues.append("Mo is dispatched before Finn and no existing Finn output is available.")
            break

    existing_mo = bool(read_pipeline("mo"))
    terminal_agents = {"quinn", "rex"}
    if any(agent in terminal_agents for agent in agents) and "mo" not in agents and not existing_mo:
        issues.append("Final QC/report agent is dispatched before Mo output is available.")

    return issues


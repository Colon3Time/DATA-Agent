from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


ANNA_OUTPUT_CONTRACT = """

---
## Anna Output Contract v3 (guided default)

When the user request requires agent work, Anna MUST follow this contract:

0. Guided coordination:
   - Default mode is guided: dispatch exactly one agent at a time, then wait for the system gate/summary before deciding the next agent.
   - Do not emit a full Scout→Dana→Eddie→Finn→Mo chain unless the user explicitly uses /run-all or strict_pipeline mode.
   - Anna is a thin coordinator: choose the next useful agent from current context; the agent does the analysis, while gates validate output after it runs.

1. Select or create a project before the first DISPATCH.
   - New task: emit CREATE_DIR for projects/<short_project_name>/input and projects/<short_project_name>/output.
   - Existing task: clearly reference the existing project name.
   - Never dispatch agents while project is unknown.
   - Every CREATE_DIR tag for a new project must appear before the first DISPATCH tag.

2. DISPATCH tags must contain one valid JSON object only:
   <DISPATCH>{"agent":"scout","task":"specific task with input/output context"}</DISPATCH>
   - Keep the whole JSON object on one line.
   - The task value must be a single-line JSON string: no literal newline characters and no escaped \\n sequences.
   - Put long instructions in one sentence separated by semicolons, commas, or numbered clauses.

3. Required DISPATCH fields:
   - agent: one of scout, dana, eddie, iris_eda, max, finn, mo, iris, vera, quinn, rex
   - task: concrete instruction with file paths, handoff context, or exact expected output

4. Optional DISPATCH fields:
   - discover: boolean true/false only
   - parallel_group: short string only, and only when agents can run safely in parallel

5. CRISP-DM guardrails:
   - If no dataset exists, dispatch scout first.
   - If dataset exists but no cleaning/EDA exists, prefer dana and/or eddie before modeling.
   - If valid Finn output already exists, treat Finn as completed and do not force Dana/Eddie reruns before Mo Phase 2.
   - Do not dispatch mo unless Finn runs before Mo in the same plan, or valid Finn output already exists.
   - Do not dispatch quinn/rex final work before Mo has produced model output, unless user explicitly asks for reporting only.
   - If required context is missing, use READ_FILE or ASK_USER before dispatching.
   - Session Memory is advisory only. Current dispatch task, active project, latest logs, and current output files override older memory.
   - Every task must name the current project/input/output context clearly enough that the agent does not infer from old projects.
   - If Quinn previously requested RESTART_CYCLE, downstream Iris/Vera/Rex may still be dispatched for failure-aware summaries, visual diagnostics, or executive issue reports. They must not present the cycle as successful until the restart is fixed.

6. Avoid low-value dispatch:
   - Do not dispatch agents for greetings, explanations, or normal chat.
   - Do not emit vague tasks such as "...", "analyze this", "ทำต่อ", or "วิเคราะห์" without context.
7. Ordering rule for cleanup/EDA:
   - If Dana output is not already available, dispatch Dana before Eddie.
   - Never place Eddie as the first cleanup/EDA dispatch when Dana has not completed.
"""


ReadPipelineFn = Callable[[str], str]


VALID_AGENTS = {"scout", "dana", "eddie", "iris_eda", "max", "finn", "mo", "iris", "vera", "quinn", "rex"}
REQUIRED_FIELDS = {"agent", "task"}
OPTIONAL_FIELDS = {"discover", "parallel_group"}
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
VAGUE_TASKS = {"...", "analyze this", "ทำต่อ", "วิเคราะห์", "analyze", "continue"}
MIN_TASK_LENGTH = 12
DATA_EXTENSIONS = (".csv", ".parquet", ".xlsx", ".xls", ".sqlite", ".db", ".json", ".md")


def _task_has_explicit_input_path(task: str) -> bool:
    text = task.strip().strip("\"'")
    if not text:
        return False
    lowered = text.lower()
    if not any(ext in lowered for ext in DATA_EXTENSIONS):
        return False
    path_match = re.search(
        r"([A-Za-z]:\\[^\s\r\n\"<>|]+(?:\.csv|\.parquet|\.xlsx|\.xls|\.sqlite|\.db|\.json|\.md)|(?:\.{1,2}\\|\.{1,2}/|projects[\\/])[^\s\"\r\n<>|]+(?:\.csv|\.parquet|\.xlsx|\.xls|\.sqlite|\.db|\.json|\.md))",
        text,
        re.IGNORECASE,
    )
    if not path_match:
        return False
    candidate = path_match.group(1).strip().strip(".,;)")
    return Path(candidate).exists()


def validate_dispatch_plan(
    dispatches: list[dict],
    *,
    active_project: Path | None,
    read_pipeline: ReadPipelineFn,
    source_text: str = "",
    mode: str = "strict",
) -> list[str]:
    """Return issues that should force Anna to repair the plan before execution."""
    issues: list[str] = []
    if not dispatches:
        return issues

    if active_project is None:
        issues.append("No active project was selected or created before DISPATCH.")

    agents = [str(d.get("agent", "")).lower() for d in dispatches]
    tasks = [str(d.get("task", "")).strip() for d in dispatches]
    explicit_input = [_task_has_explicit_input_path(task) for task in tasks]
    mode_l = (mode or "strict").strip().lower()
    if mode_l in {"guided", "guide"} and len(dispatches) > 1:
        issues.append(
            "Guided mode allows only one DISPATCH at a time; run one agent, inspect gate/summary, then decide the next agent."
        )

    if source_text:
        first_dispatch = source_text.find("<DISPATCH>")
        first_create_dir = source_text.find("<CREATE_DIR")
        if first_create_dir >= 0 and first_dispatch >= 0 and first_create_dir > first_dispatch:
            issues.append("CREATE_DIR appears after DISPATCH; create project directories before dispatching agents.")

    for idx, dispatch in enumerate(dispatches):
        keys = set(dispatch)
        missing = REQUIRED_FIELDS - keys
        extra = keys - ALLOWED_FIELDS
        if missing:
            issues.append(f"Dispatch #{idx + 1} is missing required field(s): {', '.join(sorted(missing))}.")
        if extra:
            issues.append(f"Dispatch #{idx + 1} uses unsupported field(s): {', '.join(sorted(extra))}.")
        if "discover" in dispatch and not isinstance(dispatch["discover"], bool):
            issues.append(f"Dispatch #{idx + 1} field 'discover' must be boolean true/false.")
        if "parallel_group" in dispatch:
            parallel_group = dispatch["parallel_group"]
            if not isinstance(parallel_group, str) or not parallel_group.strip() or "\n" in parallel_group or "\r" in parallel_group:
                issues.append(f"Dispatch #{idx + 1} field 'parallel_group' must be a short single-line string.")

    for idx, agent in enumerate(agents):
        if agent not in VALID_AGENTS:
            issues.append(f"Dispatch #{idx + 1} uses invalid agent '{agent}'.")
    for idx, task in enumerate(tasks):
        if len(task) < MIN_TASK_LENGTH or task.lower() in VAGUE_TASKS:
            issues.append(f"Dispatch #{idx + 1} has a vague or placeholder task.")
        if "\n" in task or "\r" in task:
            issues.append(f"Dispatch #{idx + 1} task must be a single-line JSON string without newline escapes.")

    existing_dana = bool(read_pipeline("dana"))
    existing_eddie = bool(read_pipeline("eddie"))
    existing_finn = bool(read_pipeline("finn"))
    has_dana_before = existing_dana
    has_eddie_before = existing_eddie
    has_finn_before_mo = False
    for idx, agent in enumerate(agents):
        if explicit_input[idx]:
            if agent in {"dana", "eddie", "iris_eda", "max", "finn", "mo", "iris", "vera", "quinn", "rex"}:
                if agent == "dana":
                    has_dana_before = True
                if agent == "eddie":
                    has_eddie_before = True
                if agent == "finn":
                    has_finn_before_mo = True
                continue
        if agent == "dana":
            has_dana_before = True
        if agent == "eddie":
            if not has_dana_before:
                issues.append("Eddie is dispatched before Dana and no existing Dana output is available.")
            has_eddie_before = True
        if agent == "finn":
            if not existing_finn:
                if not has_dana_before:
                    issues.append("Finn is dispatched before Dana and no existing Dana output is available.")
                if not has_eddie_before:
                    issues.append("Finn is dispatched before Eddie and no existing Eddie output is available.")
            has_finn_before_mo = True
        if agent == "mo" and not (has_finn_before_mo or existing_finn):
            issues.append("Mo is dispatched before Finn and no existing Finn output is available.")
            break

    existing_mo = bool(read_pipeline("mo"))
    terminal_agents = {"quinn", "rex"}
    if any(agent in terminal_agents and not explicit_input[idx] for idx, agent in enumerate(agents)) and "mo" not in agents and not existing_mo:
        issues.append("Final QC/report agent is dispatched before Mo output is available.")

    return issues

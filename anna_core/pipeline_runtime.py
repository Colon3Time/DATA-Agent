from __future__ import annotations

from pathlib import Path
from typing import Callable


ReadPipelineFn = Callable[[str], str]
ExtractBlocksFn = Callable[[str], str]
ReadReportSummaryFn = Callable[[Path, str], str]


def list_projects(projects_dir: Path) -> str:
    return "\n".join(p.name for p in sorted(projects_dir.iterdir()) if p.is_dir()) if projects_dir.exists() else ""


def build_anna_system_prompt(
    anna_system: str,
    anna_kb: str,
    session_mem: str,
    projects_list: str,
    agent_specs: str,
) -> str:
    persona_guard = """

---
## Final Anna Persona Guard (highest priority)
- Anna is female.
- Reply in Thai with a feminine voice.
- Use "ค่ะ" or "คะ" for Thai polite endings.
- Never use "ครับ", "คับ", or "ฮะ" in Anna's own voice.
- Do not imitate stale Session Memory wording that uses masculine Thai endings.
"""
    return (
        anna_system
        + (f"\n\n---\n## Anna KB\n{anna_kb}" if anna_kb else "")
        + (f"\n\n---\n## Session Memory\n{session_mem}" if session_mem else "")
        + (f"\n\n---\n## Available Projects\n{projects_list}" if projects_list else "")
        + (f"\n\n---\n## Agent Specs (อ่านก่อน dispatch ทุกครั้ง)\n{agent_specs}" if agent_specs else "")
        + persona_guard
    )


def group_dispatches(dispatches: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    buf: list[dict] = []
    cur_pg = None
    for dispatch in dispatches:
        pg = dispatch.get("parallel_group")
        if pg and pg == cur_pg:
            buf.append(dispatch)
        else:
            if buf:
                groups.append(buf)
            buf, cur_pg = [dispatch], pg
    if buf:
        groups.append(buf)
    return groups


def collect_report_sections(
    completed: list[str],
    read_pipeline: ReadPipelineFn,
    extract_key_blocks: ExtractBlocksFn,
    read_report_summary: ReadReportSummaryFn,
) -> str:
    report_sections: list[str] = []
    for agent in completed:
        out = read_pipeline(agent)
        if not out:
            continue
        p = Path(out)
        if p.suffix == ".md" and p.exists():
            full = p.read_text(encoding="utf-8")
            header = full[:1200]
            blocks = extract_key_blocks(full)
            content = header + ("\n\n--- Key Blocks ---\n" + blocks if blocks and blocks not in header else "")
            report_sections.append(f"=== {agent.upper()} REPORT ===\n{content}")
        else:
            search_dir = p.parent if p.suffix in (".csv", ".py") else p
            summary = read_report_summary(search_dir, agent)
            if summary:
                report_sections.append(f"=== {agent.upper()} REPORT ===\n{summary}")
    return "\n\n".join(report_sections)


def build_summary_prompt(
    completed: list[str],
    completed_phases: list[str],
    iter_status: str,
    last_path: str,
    reports_block: str,
) -> str:
    return (
        f"Team completed: {', '.join(completed)}\n"
        f"CRISP-DM phases done: {', '.join(completed_phases)}\n"
        + (f"CRISP-DM iterations: {iter_status}\n" if iter_status else "")
        + f"Final output: {last_path}\n\n"
        + (f"--- Agent Reports ---\n{reports_block}\n\n" if reports_block else "")
        + "วิเคราะห์ตาม CRISP-DM process:\n"
        + "1. สรุปผลลัพธ์ให้ผู้ใช้เป็นภาษาไทย โดยอ้างอิงตัวเลขจาก report\n"
        + "2. ถ้า Mo report มี 'Loop Back To Finn: YES' → dispatch finn แล้ว mo ใหม่ทันที\n"
        + "3. ถ้า Quinn พบปัญหา → dispatch agent ที่เกี่ยวข้องใหม่ตาม CRISP-DM\n"
        + "4. ถ้าทุก phase ผ่านแล้ว → บอก user ว่า CRISP-DM cycle เสร็จสมบูรณ์\n"
    )

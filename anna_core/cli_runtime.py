from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .pipeline_store import PipelineStore
from .state import OrchestratorState


@dataclass(frozen=True)
class CliPalette:
    reset: str
    bold: str
    dim: str
    cyan: str
    green: str
    yellow: str
    red: str
    blue: str
    magenta: str
    white: str


class CliRenderer:
    def __init__(
        self,
        *,
        state: OrchestratorState,
        pipeline: PipelineStore,
        projects_dir: Path,
        deepseek_model: str,
        claude_model: str,
        claude_limit: int,
        mode: str,
        palette: CliPalette,
    ) -> None:
        self.state = state
        self.pipeline = pipeline
        self.projects_dir = projects_dir
        self.deepseek_model = deepseek_model
        self.claude_model = claude_model
        self.claude_limit = claude_limit
        self.mode = mode
        self.p = palette

    def print_header(self) -> None:
        p = self.p
        ds_ok = bool(os.environ.get("DEEPSEEK_API_KEY"))
        cl_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
        plain = [
            "DataScienceOS  —  Anna (CEO)",
            f"DeepSeek: {self.deepseek_model}  |  Claude: {self.claude_model}",
            f"PATH-BASED pipeline v2  |  mode: {self.mode}  |  Claude limit: {self.claude_limit}",
            "Type  help  for commands",
        ]
        width = max(len(line) for line in plain) + 4

        def box_row(plain_text: str, colored_text: str) -> str:
            pad = " " * (width - 2 - len(plain_text))
            return f"{p.cyan}│{p.reset}  {colored_text}{pad}  {p.cyan}│{p.reset}"

        print(f"{p.cyan}┌{'─'*width}┐{p.reset}")
        print(box_row(plain[0], f"{p.bold}{p.white}DataScienceOS{p.reset}  {p.dim}—{p.reset}  {p.bold}{p.yellow}Anna (CEO){p.reset}"))
        print(box_row(plain[1], f"{p.blue}DeepSeek:{p.reset} {p.bold}{self.deepseek_model}{p.reset}  {p.dim}|{p.reset}  {p.magenta}Claude:{p.reset} {p.bold}{self.claude_model}{p.reset}"))
        print(box_row(plain[2], f"{p.dim}PATH-BASED pipeline v2{p.reset}  {p.dim}|{p.reset}  mode: {p.bold}{self.mode}{p.reset}  {p.dim}|{p.reset}  Claude limit: {p.magenta}{p.bold}{self.claude_limit}{p.reset}"))
        print(box_row(plain[3], f"Type  {p.bold}{p.white}help{p.reset}  for commands"))
        print(f"{p.cyan}└{'─'*width}┘{p.reset}")
        print()

        ds_str = f"{p.green}✓{p.reset}" if ds_ok else f"{p.red}✗ ไม่พบ key{p.reset}"
        cl_str = f"{p.green}✓{p.reset}" if cl_ok else f"{p.red}✗ ไม่พบ key{p.reset}"
        print(f"  {p.blue}{p.bold}DeepSeek:{p.reset} {ds_str}    {p.magenta}{p.bold}Claude:{p.reset} {cl_str}  {p.dim}(limit: {self.claude_limit} calls/session){p.reset}")
        print()

    def print_status(self) -> None:
        p = self.p
        print(f"\n{p.cyan}┌─ STATUS ─────────────────────────────────────────────┐{p.reset}")
        project = self.state.active_project.name if self.state.active_project else "ไม่มี"
        print(f"{p.cyan}│{p.reset}  Project : {p.bold}{project}{p.reset}")
        print(f"{p.cyan}│{p.reset}  Claude  : {self.state.claude_calls}/{self.claude_limit} calls")
        if self.state.agent_iter_count:
            iters = ", ".join(f"{a}×{n}" for a, n in self.state.agent_iter_count.items() if n > 1)
            if iters:
                print(f"{p.cyan}│{p.reset}  Iterations: {iters}")
        path_files = self.pipeline.path_files()
        if path_files:
            print(f"{p.cyan}│{p.reset}  Pipeline outputs:")
            for pf in path_files:
                agent = pf.stem.replace("_path", "")
                value = pf.read_text(encoding="utf-8").strip()
                exists_mark = f"{p.green}✓{p.reset}" if Path(value).exists() else f"{p.red}✗{p.reset}"
                print(f"{p.cyan}│{p.reset}    {p.bold}{agent:<10}{p.reset} {exists_mark}  {p.dim}{value[-55:]}{p.reset}")
        else:
            print(f"{p.cyan}│{p.reset}  {p.dim}ยังไม่มี pipeline output{p.reset}")
        print(f"{p.cyan}└──────────────────────────────────────────────────────┘{p.reset}")

    def print_claude_usage(self) -> None:
        p = self.p
        used = self.state.claude_calls
        limit = self.claude_limit
        pct = int(used / limit * 100) if limit > 0 else 100
        bar_len = 20
        filled = int(bar_len * used / limit) if limit > 0 else bar_len
        bar_color = p.green if pct < 60 else (p.yellow if pct < 90 else p.red)
        bar = f"{bar_color}{'█' * filled}{p.dim}{'░' * (bar_len - filled)}{p.reset}"
        print(f"\n  {p.magenta}{p.bold}Claude usage:{p.reset}  {bar}  {p.bold}{used}/{limit}{p.reset}  ({pct}%)")
        if used >= limit:
            print(f"  {p.red}  ✗ ถึง limit แล้ว — ทุก call จะใช้ DeepSeek แทน{p.reset}")
        else:
            print(f"  เหลืออีก {p.bold}{limit - used}{p.reset} calls  →  reset ด้วย {p.bold}end session{p.reset}  หรือ {p.bold}--claude-limit N{p.reset}")
        print()

    def print_help(self) -> None:
        p = self.p
        print(f"""
{p.cyan}┌─ คำสั่ง ──────────────────────────────────────────────┐{p.reset}
{p.cyan}│{p.reset}  {p.bold}{p.white}<ข้อความ>{p.reset}            {p.yellow}»{p.reset} Anna รับ แล้ว pipeline อัตโนมัติ
{p.cyan}│{p.reset}  {p.bold}{p.white}!! <ข้อความ>{p.reset}          {p.yellow}»{p.reset} {p.magenta}Claude{p.reset} discover mode
{p.cyan}│{p.reset}  {p.bold}{p.white}@<agent> <task>{p.reset}       {p.yellow}»{p.reset} dispatch ตรงไป agent ({p.blue}DeepSeek{p.reset})
{p.cyan}│{p.reset}  {p.bold}{p.white}@<agent>! <task>{p.reset}      {p.yellow}»{p.reset} dispatch ตรงไป agent ({p.magenta}Claude{p.reset})
{p.cyan}│{p.reset}  {p.bold}{p.white}project <name>{p.reset}        {p.yellow}»{p.reset} set active project
{p.cyan}│{p.reset}  {p.bold}{p.white}resume <name>{p.reset}         {p.yellow}»{p.reset} ต่อ pipeline ที่ค้างไว้
{p.cyan}│{p.reset}  {p.bold}{p.white}status{p.reset}                {p.yellow}»{p.reset} ดู pipeline output + Claude usage
{p.cyan}│{p.reset}  {p.bold}{p.white}kb <agent>{p.reset}            {p.yellow}»{p.reset} ดู knowledge base ของ agent
{p.cyan}│{p.reset}  {p.bold}{p.white}claude{p.reset}                {p.yellow}»{p.reset} ดู {p.magenta}Claude{p.reset} usage / calls เหลือ
{p.cyan}│{p.reset}  {p.bold}{p.white}end session{p.reset}           {p.yellow}»{p.reset} ล้าง history + reset Claude calls
{p.cyan}│{p.reset}  {p.bold}{p.white}exit{p.reset}                  {p.yellow}»{p.reset} ออกจากระบบ
{p.cyan}│{p.reset}  {p.dim}--claude-limit N{p.reset}      {p.yellow}»{p.reset} {p.dim}ตั้ง limit เมื่อเริ่มโปรแกรม (default 10){p.reset}
{p.cyan}└──────────────────────────────────────────────────────┘{p.reset}""")

    def print_kb(self, name: str, content: str) -> None:
        p = self.p
        if content:
            bar = "─" * max(0, 44 - len(name))
            print(f"\n{p.cyan}┌─ KB: {p.bold}{name}{p.reset}{p.cyan} {bar}┐{p.reset}")
            print(content)
            print(f"{p.cyan}└{'─'*50}┘{p.reset}")
        else:
            print(f"{p.yellow}  [{name}]{p.reset} ยังไม่มี Knowledge Base")

    def resolve_project(self, name: str) -> tuple[Path | None, str]:
        project = self.projects_dir / name
        if project.exists():
            return project, ""
        matches = [d for d in self.projects_dir.iterdir() if d.is_dir() and name.lower() in d.name.lower()] if self.projects_dir.exists() else []
        if len(matches) == 1:
            return matches[0], ""
        if len(matches) > 1:
            return None, f"พบหลาย project: {', '.join(m.name for m in matches)}"
        return None, f"ไม่พบ project: {name}"

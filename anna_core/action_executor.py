from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .actions import WorkspacePaths
from .runner import run_inline_python, run_shell_command


LogFn = Callable[[str, str, str, str], None]
SaveKbFn = Callable[[str, str, str], None]
AskDeepseekFn = Callable[[str, str, str], str]
AskClaudeFn = Callable[[str, str, str], str]
SetActiveProjectFn = Callable[[Path], None]


@dataclass(frozen=True)
class ActionPalette:
    reset: str
    bold: str
    dim: str
    cyan: str
    green: str
    red: str
    blue: str
    magenta: str


class ActionExecutor:
    """Execute Anna action tags and return compact action results."""

    def __init__(
        self,
        *,
        base_dir: Path,
        projects_dir: Path,
        workspace_paths: WorkspacePaths,
        log: LogFn,
        save_kb: SaveKbFn,
        ask_deepseek: AskDeepseekFn,
        ask_claude: AskClaudeFn,
        set_active_project: SetActiveProjectFn,
        palette: ActionPalette,
    ) -> None:
        self.base_dir = base_dir
        self.projects_dir = projects_dir
        self.workspace_paths = workspace_paths
        self.log = log
        self.save_kb = save_kb
        self.ask_deepseek = ask_deepseek
        self.ask_claude = ask_claude
        self.set_active_project = set_active_project
        self.p = palette

    def execute(self, response: str) -> str:
        parts: list[str] = []
        self._read_file(response, parts)
        self._run_shell(response, parts)
        self._write_file(response, parts)
        self._append_file(response, parts)
        self._edit_file(response, parts)
        self._create_dir(response, parts)
        self._delete_file(response, parts)
        self._update_kb(response, parts)
        self._ask_deepseek(response, parts)
        self._ask_claude(response, parts)
        self._research(response, parts)
        self._run_python(response, parts)
        return "\n\n".join(parts)

    def _read_file(self, response: str, parts: list[str]) -> None:
        for match in re.finditer(r'<READ_FILE\s+path="([^"]+)"\s*/?>', response):
            raw = match.group(1)
            fpath = self.workspace_paths.resolve(raw)
            print(f"\n{self.p.cyan}  ▶ READ_FILE{self.p.reset}  {self.p.dim}{raw}{self.p.reset}")
            self.log("anna", f"READ_FILE: {raw}", "full-power", "")
            try:
                content = fpath.read_text(encoding="utf-8")
                parts.append(f"[READ_FILE: {raw}]\n{content[:100000]}")
            except Exception as e:
                parts.append(f"[READ_FILE ERROR: {e}]")
                self.log("anna", f"READ_FILE ERROR: {raw} — {e}", "full-power", "")

    def _run_shell(self, response: str, parts: list[str]) -> None:
        for match in re.finditer(r"<RUN_SHELL>(.*?)</RUN_SHELL>", response, re.DOTALL):
            cmd = match.group(1).strip()
            print(f"\n{self.p.cyan}  ▶ RUN_SHELL{self.p.reset}  {self.p.dim}{cmd[:60]}{self.p.reset}")
            self.log("anna", f"RUN_SHELL: {cmd[:100]}", "full-power", "")
            try:
                result = run_shell_command(cmd, self.base_dir, timeout_seconds=60)
                out = result.combined_output[:1500]
                parts.append(f"[RUN_SHELL: {cmd[:60]}]\n{out}")
                if result.returncode != 0:
                    self.log("anna", f"RUN_SHELL exit={result.returncode}: {cmd[:60]}", "full-power", "")
            except Exception as e:
                parts.append(f"[RUN_SHELL ERROR: {e}]")
                self.log("anna", f"RUN_SHELL ERROR: {cmd[:60]} — {e}", "full-power", "")

    def _write_file(self, response: str, parts: list[str]) -> None:
        for match in re.finditer(r'<WRITE_FILE\s+path="([^"]+)">(.*?)</WRITE_FILE>', response, re.DOTALL):
            raw_path, content = match.group(1), match.group(2)
            fpath = self.workspace_paths.resolve(raw_path)
            print(f"\n{self.p.green}  ▶ WRITE_FILE{self.p.reset}  {self.p.dim}{raw_path}{self.p.reset}")
            try:
                fpath.parent.mkdir(parents=True, exist_ok=True)
                if fpath.suffix == ".py":
                    content = re.sub(r"^\s*```(?:python|py)?\s*\r?\n", "", content.strip(), flags=re.IGNORECASE)
                    content = re.sub(r"\r?\n```\s*$", "\n", content)
                fpath.write_text(content, encoding="utf-8")
                parts.append(f"[WRITE_FILE: {raw_path}] เขียนสำเร็จ")
                self.log("anna", f"WRITE_FILE: {raw_path}", "full-power", "")
            except Exception as e:
                parts.append(f"[WRITE_FILE ERROR: {e}]")
                self.log("anna", f"WRITE_FILE ERROR: {raw_path} — {e}", "full-power", "")

    def _append_file(self, response: str, parts: list[str]) -> None:
        for match in re.finditer(r'<APPEND_FILE\s+path="([^"]+)">(.*?)</APPEND_FILE>', response, re.DOTALL):
            raw_path, content = match.group(1), match.group(2)
            fpath = self.workspace_paths.resolve(raw_path)
            print(f"\n{self.p.green}  ▶ APPEND_FILE{self.p.reset}  {self.p.dim}{raw_path}{self.p.reset}")
            try:
                fpath.parent.mkdir(parents=True, exist_ok=True)
                with open(fpath, "a", encoding="utf-8") as fp:
                    fp.write(content)
                parts.append(f"[APPEND_FILE: {raw_path}] เพิ่มสำเร็จ")
                self.log("anna", f"APPEND_FILE: {raw_path}", "full-power", "")
            except Exception as e:
                parts.append(f"[APPEND_FILE ERROR: {e}]")
                self.log("anna", f"APPEND_FILE ERROR: {raw_path} — {e}", "full-power", "")

    def _edit_file(self, response: str, parts: list[str]) -> None:
        pattern = r'<EDIT_FILE\s+path="([^"]+)"><old>(.*?)</old><new>(.*?)</new></EDIT_FILE>'
        for match in re.finditer(pattern, response, re.DOTALL):
            raw_path, old, new = match.group(1), match.group(2), match.group(3)
            fpath = self.workspace_paths.resolve(raw_path)
            print(f"\n{self.p.green}  ▶ EDIT_FILE{self.p.reset}  {self.p.dim}{raw_path}{self.p.reset}")
            try:
                original = fpath.read_text(encoding="utf-8")
                fpath.write_text(original.replace(old, new, 1), encoding="utf-8")
                parts.append(f"[EDIT_FILE: {raw_path}] แก้ไขสำเร็จ")
                self.log("anna", f"EDIT_FILE: {raw_path}", "full-power", "")
            except Exception as e:
                parts.append(f"[EDIT_FILE ERROR: {e}]")
                self.log("anna", f"EDIT_FILE ERROR: {raw_path} — {e}", "full-power", "")

    def _create_dir(self, response: str, parts: list[str]) -> None:
        for match in re.finditer(r'<CREATE_DIR\s+path="([^"]+)"\s*/?>', response):
            raw_path = match.group(1)
            dpath = self.workspace_paths.resolve(raw_path)
            print(f"\n{self.p.green}  ▶ CREATE_DIR{self.p.reset}  {self.p.dim}{raw_path}{self.p.reset}")
            try:
                dpath.mkdir(parents=True, exist_ok=True)
                parts.append(f"[CREATE_DIR: {raw_path}] สร้างสำเร็จ")
                self.log("anna", f"CREATE_DIR: {raw_path}", "full-power", "")
                try:
                    rel_parts = dpath.relative_to(self.projects_dir).parts
                    if rel_parts:
                        self.set_active_project(self.projects_dir / rel_parts[0])
                except ValueError:
                    pass
            except Exception as e:
                parts.append(f"[CREATE_DIR ERROR: {e}]")

    def _delete_file(self, response: str, parts: list[str]) -> None:
        for match in re.finditer(r'<DELETE_FILE\s+path="([^"]+)"\s*/?>', response):
            raw_path = match.group(1)
            print(f"\n{self.p.red}  ▶ DELETE_FILE{self.p.reset}  {self.p.dim}{raw_path}{self.p.reset}")
            try:
                fpath = self.workspace_paths.resolve_for_delete(raw_path)
                fpath.unlink()
                parts.append(f"[DELETE_FILE: {raw_path}] ลบสำเร็จ")
                self.log("anna", f"DELETE_FILE: {raw_path}", "full-power", "")
            except Exception as e:
                parts.append(f"[DELETE_FILE ERROR: {e}]")

    def _update_kb(self, response: str, parts: list[str]) -> None:
        for match in re.finditer(r'<UPDATE_KB\s+agent="([^"]+)">(.*?)</UPDATE_KB>', response, re.DOTALL):
            agent, content = match.group(1), match.group(2).strip()
            self.save_kb(agent, content, "feedback")
            print(f"\n{self.p.green}  ▶ UPDATE_KB{self.p.reset}  agent={self.p.bold}{agent}{self.p.reset}")
            parts.append(f"[UPDATE_KB: {agent}] อัปเดตสำเร็จ")

    def _ask_deepseek(self, response: str, parts: list[str]) -> None:
        for match in re.finditer(r"<ASK_DEEPSEEK>(.*?)</ASK_DEEPSEEK>", response, re.DOTALL):
            q = match.group(1).strip()
            print(f"\n{self.p.blue}  ▶ ASK_DEEPSEEK{self.p.reset}")
            self.log("anna", f"ASK_DEEPSEEK: {q[:80]}", "full-power", "")
            ans = self.ask_deepseek("You are a helpful AI assistant.", q, "DEEPSEEK direct")
            parts.append(f"[ASK_DEEPSEEK]\nQ: {q[:200]}\nA: {ans[:1000]}")

    def _ask_claude(self, response: str, parts: list[str]) -> None:
        for match in re.finditer(r"<ASK_CLAUDE>(.*?)</ASK_CLAUDE>", response, re.DOTALL):
            q = match.group(1).strip()
            print(f"\n{self.p.magenta}  ▶ ASK_CLAUDE{self.p.reset}")
            self.log("anna", f"ASK_CLAUDE: {q[:80]}", "full-power", "")
            ans = self.ask_claude("You are a helpful AI assistant.", q, "CLAUDE direct")
            parts.append(f"[ASK_CLAUDE]\nQ: {q[:200]}\nA: {ans[:1000]}")

    def _research(self, response: str, parts: list[str]) -> None:
        for match in re.finditer(r"<RESEARCH>(.*?)</RESEARCH>", response, re.DOTALL):
            topic = match.group(1).strip()
            print(f"\n{self.p.blue}  ▶ RESEARCH{self.p.reset}  {self.p.dim}{topic[:60]}{self.p.reset}")
            self.log("anna", f"RESEARCH: {topic[:80]}", "full-power", "")
            ans = self.ask_deepseek("You are a research assistant. Be thorough.", topic, "RESEARCH")
            first_finding = ans.strip().split("\n\n")[0][:400]
            self.save_kb("anna", f"Research: {topic[:100]}\nKey finding: {first_finding}", "discovery")
            parts.append(f"[RESEARCH: {topic[:60]}]\n{ans[:1000]}")

    def _run_python(self, response: str, parts: list[str]) -> None:
        for match in re.finditer(r"<RUN_PYTHON>(.*?)</RUN_PYTHON>", response, re.DOTALL):
            code = match.group(1).strip()
            preview = code[:60].replace("\n", " ")
            print(f"\n{self.p.cyan}  ▶ RUN_PYTHON{self.p.reset}  {self.p.dim}{preview}{self.p.reset}")
            self.log("anna", f"RUN_PYTHON: {code[:80]}", "full-power", "")
            try:
                result = run_inline_python(code, self.base_dir, timeout_seconds=30)
                out = result.combined_output[:1500]
                parts.append(f"[RUN_PYTHON]\n{out}")
                if result.returncode != 0:
                    self.log("anna", f"RUN_PYTHON error exit={result.returncode}: {result.stderr[:100]}", "full-power", "")
            except Exception as e:
                parts.append(f"[RUN_PYTHON ERROR: {e}]")
                self.log("anna", f"RUN_PYTHON ERROR: {e}", "full-power", "")

"""
DeepSeek Direct Chat — Enhanced UI with Rich
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
from anna_core.env import load_app_env

# UI Libraries
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style as PTStyle

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
from rich.theme import Theme

# Setup Rich Console
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "error": "bold red",
    "user": "bold cyan",
    "ai": "bold green",
    "system": "italic yellow"
})
console = Console(theme=custom_theme)

load_app_env(r"D:\DATA-ScinceOS\.env")

DEEPSEEK_URL   = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

history     = []
input_history = InMemoryHistory()

# prompt_toolkit style for the input line
prompt_style = PTStyle.from_dict({
    "prompt": "ansicyan bold",
})


def chat(user_message: str, system: str = "") -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        console.print("[error]ไม่พบ DEEPSEEK_API_KEY ใน .env[/error]")
        return ""

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = requests.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": DEEPSEEK_MODEL, "messages": messages, "stream": True},
            stream=True, timeout=180,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        console.print("[error]เชื่อมต่อ DeepSeek ไม่ได้[/error]")
        return ""
    except requests.exceptions.Timeout:
        console.print("[error]DeepSeek timeout[/error]")
        return ""
    except Exception as e:
        console.print(f"[error]Error: {str(e)}[/error]")
        return ""

    full_content = ""
    
    console.print("\n[ai]DeepSeek:[/ai]")
    
    with Live(Text(""), console=console, refresh_per_second=10) as live:
        for line in response.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if text.startswith("data: "):
                text = text[6:]
            if text == "[DONE]":
                break
            try:
                token = json.loads(text)["choices"][0]["delta"].get("content", "")
                full_content += token
                live.update(Markdown(full_content))
            except (json.JSONDecodeError, KeyError):
                pass
    
    print() # newline after response
    return full_content


def save_log(role: str, content: str):
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    f = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}_deepseek_direct.md"
    ts = datetime.now().strftime("%H:%M")
    with open(f, "a", encoding="utf-8") as fp:
        fp.write(f"[{ts}] {role}: {content[:300]}\n")


HELP_TEXT = """
[bold cyan]Commands:[/bold cyan]
  [yellow]/clear[/yellow]     → ล้าง history เริ่มใหม่
  [yellow]/history[/yellow]   → ดู history ปัจจุบัน
  [yellow]/system[/yellow]    → set system prompt ใหม่
  [yellow]/exit[/yellow]      → ออกจากโปรแกรม
  [dim]↑ / ↓       → เลื่อนดูข้อความเก่า[/dim]
"""


def main():
    system_prompt = ""

    # Welcome Banner
    banner = Panel.fit(
        "[bold cyan]DeepSeek Direct Chat[/bold cyan]\n"
        f"[dim]Model: {DEEPSEEK_MODEL}[/dim]\n"
        "[italic]Type /help for more info[/italic]",
        border_style="cyan"
    )
    console.print(banner)

    api_ok = bool(os.environ.get("DEEPSEEK_API_KEY"))
    status_msg = "[bold green]✓ พร้อมใช้งาน[/bold green]" if api_ok else "[bold red]✗ ไม่พบ DEEPSEEK_API_KEY[/bold red]"
    console.print(f"Status: {status_msg}\n")

    while True:
        try:
            # Using prompt_toolkit for history support
            user_input = prompt(
                "คุณ: ",
                history=input_history,
                style=prompt_style,
            ).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[info]ปิดโปรแกรม...[/info]")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            break
        if user_input == "/help":
            console.print(Panel(HELP_TEXT, title="Help", border_style="yellow"))
            continue
        if user_input == "/clear":
            history.clear()
            console.print("[info]ล้าง history เรียบร้อย[/info]\n")
            continue
        if user_input == "/history":
            if not history:
                console.print("[info]ยังไม่มีประวัติการสนทนา[/info]\n")
            else:
                for h in history:
                    role_style = "cyan" if h['role'] == 'user' else "green"
                    console.print(f"  [[{role_style}]{h['role']}[/{role_style}]] {h['content'][:100]}...")
            print()
            continue
        if user_input.startswith("/system"):
            system_prompt = user_input[7:].strip()
            console.print(f"[system]System prompt: {system_prompt[:80] or '(ล้างแล้ว)'}[/system]\n")
            continue

        save_log("User", user_input)
        
        result = chat(user_input, system=system_prompt)
        
        if result:
            save_log("DeepSeek", result)
            history.append({"role": "user",      "content": user_input})
            history.append({"role": "assistant", "content": result})


if __name__ == "__main__":
    main()

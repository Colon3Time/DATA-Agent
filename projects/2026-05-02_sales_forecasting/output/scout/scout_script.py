import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


COMPETITION = "walmart-recruiting-store-sales-forecasting"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def kaggle_credentials_status() -> tuple[bool, str]:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True, "KAGGLE_USERNAME/KAGGLE_KEY are set"
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            data = json.loads(kaggle_json.read_text(encoding="utf-8"))
            if data.get("username") and data.get("key"):
                return True, f"{kaggle_json} exists"
        except Exception as exc:
            return False, f"{kaggle_json} exists but is not readable JSON: {exc}"
    return False, "Missing KAGGLE_USERNAME/KAGGLE_KEY and ~/.kaggle/kaggle.json"


def profile_csv(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    profile = [
        "# DATASET_PROFILE",
        "",
        "Status: loaded",
        "Dataset: Walmart Recruiting - Store Sales Forecasting",
        f"Input CSV: {csv_path}",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        "Target Column: Weekly_Sales" if "Weekly_Sales" in df.columns else "Target Column: unknown",
        "",
        "Columns:",
    ]
    for col in df.columns:
        profile.append(f"- {col}: {df[col].dtype}, missing={int(df[col].isna().sum())}")
    write_text(output_dir / "dataset_profile.md", "\n".join(profile) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default="")
    args, _ = parser.parse_known_args()

    output_dir = Path(args.output_dir).resolve()
    project_dir = output_dir.parent.parent
    input_dir = project_dir / "input"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    existing_csvs = sorted(input_dir.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if existing_csvs:
        profile_csv(existing_csvs[0], output_dir)
        write_text(
            output_dir / "scout_report.md",
            f"# Scout Dataset Brief\n\nStatus: dataset already exists in input.\nInput CSV: {existing_csvs[0]}\n",
        )
        print(f"[STATUS] Found existing dataset: {existing_csvs[0]}")
        return 0

    creds_ok, creds_msg = kaggle_credentials_status()
    modules = {name: has_module(name) for name in ("kagglehub", "kaggle", "opendatasets")}
    blockers = []
    if not creds_ok:
        blockers.append(creds_msg)
    if not any(modules.values()):
        blockers.append("Missing Kaggle download tooling: install kagglehub, kaggle, or opendatasets")

    if blockers:
        repair = [
            "# Scout Repair",
            "",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Project: {project_dir.name}",
            f"Requested dataset: {COMPETITION}",
            "",
            "Status: BLOCKED",
            "",
            "Blockers:",
            *[f"- {item}" for item in blockers],
            "",
            "Verified:",
            f"- input_dir: {input_dir}",
            "- input CSV files: none",
            f"- modules: {modules}",
            "",
            "NEED_CLAUDE: Kaggle credentials/tooling are unavailable in this runtime, so Scout cannot download the real Walmart competition dataset. Do not continue with synthetic data.",
        ]
        text = "\n".join(repair) + "\n"
        write_text(output_dir / "REPAIR.md", text)
        write_text(output_dir / "scout_report.md", text)
        print("[BLOCKED] " + "; ".join(blockers))
        return 2

    try:
        if modules["kagglehub"]:
            import kagglehub

            downloaded = Path(kagglehub.competition_download(COMPETITION, path=str(input_dir)))
            print(f"[STATUS] kagglehub downloaded: {downloaded}")
        elif modules["kaggle"]:
            cmd = [
                sys.executable,
                "-m",
                "kaggle",
                "competitions",
                "download",
                "-c",
                COMPETITION,
                "-p",
                str(input_dir),
                "--unzip",
            ]
            subprocess.run(cmd, check=True)
            print("[STATUS] kaggle CLI download complete")
        else:
            import opendatasets as od

            od.download(f"https://www.kaggle.com/competitions/{COMPETITION}", data_dir=str(input_dir))
            print("[STATUS] opendatasets download complete")
    except Exception as exc:
        repair = (
            "# Scout Repair\n\n"
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Project: {project_dir.name}\n"
            f"Requested dataset: {COMPETITION}\n\n"
            "Status: BLOCKED\n\n"
            f"Download error: {exc}\n\n"
            "NEED_CLAUDE: Real Kaggle download failed. Check credentials, competition terms acceptance, and network access. Do not continue with synthetic data.\n"
        )
        write_text(output_dir / "REPAIR.md", repair)
        write_text(output_dir / "scout_report.md", repair)
        print(f"[BLOCKED] {exc}")
        return 2

    csvs = sorted(input_dir.glob("**/*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        repair = "# Scout Repair\n\nStatus: BLOCKED\n\nDownload completed but no CSV files were found in input/.\n"
        write_text(output_dir / "REPAIR.md", repair)
        write_text(output_dir / "scout_report.md", repair)
        return 2

    profile_csv(csvs[0], output_dir)
    write_text(
        output_dir / "scout_report.md",
        f"# Scout Dataset Brief\n\nStatus: loaded\nDataset: {COMPETITION}\nPrimary CSV: {csvs[0]}\n",
    )
    print(f"[STATUS] Primary CSV: {csvs[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

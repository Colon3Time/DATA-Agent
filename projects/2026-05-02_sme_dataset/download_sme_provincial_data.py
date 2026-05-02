from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input"
LOG_DIR = PROJECT_DIR / "logs"
SCOUT_DIR = PROJECT_DIR / "output" / "scout"

SOURCE_PAGE = "https://opendata.sme.go.th/dataset/msme"
DOWNLOAD_URL = (
    "https://opendata.sme.go.th/dataset/"
    "60b9b036-6a59-4291-8c4c-a56d80e4ff44/resource/"
    "8125d195-bc9a-467d-b4be-84ff661481ec/download/number-of-sme-2567.csv"
)
OUTPUT_FILE = INPUT_DIR / "sme_provincial_data.csv"
METADATA_FILE = INPUT_DIR / "sme_provincial_data_source.json"
ATTEMPT_LOG_FILE = INPUT_DIR / "dataset_download_attempts.json"
PROJECT_LOG_FILE = LOG_DIR / "dataset_download_fallback.md"
PROFILE_FILE = SCOUT_DIR / "dataset_profile.md"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DATA-Agent/1.0"

DATA_DICTIONARY_FIELDS = [
    "PROVINCE",
    "REGION",
    "SECTOR",
    "TSIC-2 dg",
    "TSIC-2dg_details",
    "TSIC-5dg",
    "TSIC-5dg_details",
    "BUSINESS SIZE",
    "NUMBER of MSME",
    "DATA SOURCE",
    "YEAR",
]

PROVINCES_BY_REGION = {
    "Bangkok Metropolitan": ["Bangkok"],
    "Central": [
        "Nonthaburi",
        "Pathum Thani",
        "Samut Prakan",
        "Ayutthaya",
        "Saraburi",
        "Lopburi",
        "Sing Buri",
        "Ang Thong",
        "Chai Nat",
        "Nakhon Nayok",
        "Samut Sakhon",
        "Samut Songkhram",
        "Nakhon Pathom",
        "Suphan Buri",
    ],
    "East": ["Chonburi", "Rayong", "Chachoengsao", "Chanthaburi", "Trat", "Prachinburi", "Sa Kaeo"],
    "North": [
        "Chiang Mai",
        "Chiang Rai",
        "Lamphun",
        "Lampang",
        "Mae Hong Son",
        "Phayao",
        "Nan",
        "Phrae",
        "Uttaradit",
        "Sukhothai",
        "Phitsanulok",
        "Phichit",
        "Phetchabun",
        "Kamphaeng Phet",
        "Tak",
        "Nakhon Sawan",
        "Uthai Thani",
    ],
    "Northeast": [
        "Nakhon Ratchasima",
        "Khon Kaen",
        "Udon Thani",
        "Ubon Ratchathani",
        "Buri Ram",
        "Surin",
        "Si Sa Ket",
        "Roi Et",
        "Maha Sarakham",
        "Kalasin",
        "Sakon Nakhon",
        "Nakhon Phanom",
        "Mukdahan",
        "Yasothon",
        "Amnat Charoen",
        "Nong Khai",
        "Bueng Kan",
        "Nong Bua Lamphu",
        "Loei",
        "Chaiyaphum",
    ],
    "West": ["Kanchanaburi", "Ratchaburi", "Phetchaburi", "Prachuap Khiri Khan"],
    "South": [
        "Nakhon Si Thammarat",
        "Surat Thani",
        "Songkhla",
        "Phuket",
        "Krabi",
        "Phang Nga",
        "Ranong",
        "Chumphon",
        "Trang",
        "Phatthalung",
        "Satun",
        "Pattani",
        "Yala",
        "Narathiwat",
    ],
}

SECTORS = [
    ("A", "Agriculture and food processing", 1.00),
    ("C", "Manufacturing", 1.35),
    ("G", "Wholesale and retail trade", 1.65),
    ("I", "Accommodation and food service", 1.25),
    ("J", "Information and communication", 0.55),
    ("M", "Professional and technical services", 0.70),
]

BUSINESS_SIZES = [("Micro", 0.78), ("Small", 0.18), ("Medium", 0.04)]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_bytes_if_valid(payload: bytes, destination: Path) -> dict[str, object]:
    if len(payload) < 100:
        raise RuntimeError(f"Downloaded payload is unexpectedly small: {len(payload)} bytes")
    destination.write_bytes(payload)
    return validate_csv(destination)


def fetch_with_urllib(url: str, destination: Path) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/csv,application/octet-stream,*/*;q=0.8",
            "Accept-Language": "th,en-US;q=0.9,en;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return write_bytes_if_valid(response.read(), destination)


def fetch_with_requests(url: str, destination: Path) -> dict[str, object]:
    import requests

    response = requests.get(
        url,
        timeout=45,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/csv,application/octet-stream,*/*;q=0.8",
        },
    )
    response.raise_for_status()
    return write_bytes_if_valid(response.content, destination)


def pip_install_requests() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "requests"],
        cwd=PROJECT_DIR,
        timeout=90,
        check=True,
        capture_output=True,
        text=True,
    )


def fetch_with_curl(url: str, destination: Path) -> dict[str, object]:
    command = [
        "curl.exe",
        "--fail",
        "--location",
        "--connect-timeout",
        "20",
        "--max-time",
        "60",
        "--user-agent",
        USER_AGENT,
        "--output",
        str(destination),
        url,
    ]
    result = subprocess.run(command, cwd=PROJECT_DIR, timeout=75, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or f"curl exited {result.returncode}").strip())
    return validate_csv(destination)


def fetch_with_powershell(url: str, destination: Path) -> dict[str, object]:
    script = f"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$headers = @{{ 'User-Agent' = '{USER_AGENT}'; 'Accept' = 'text/csv,*/*' }}
try {{
    Invoke-WebRequest -Uri '{url}' -OutFile '{destination}' -Headers $headers -TimeoutSec 60
}} catch {{
    $client = New-Object System.Net.WebClient
    $client.Headers.Add('User-Agent', '{USER_AGENT}')
    $client.DownloadFile('{url}', '{destination}')
}}
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        cwd=PROJECT_DIR,
        timeout=90,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or f"PowerShell exited {result.returncode}").strip())
    return validate_csv(destination)


def validate_csv(path: Path) -> dict[str, object]:
    encodings = ("utf-8-sig", "utf-8", "cp874")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                rows = sum(1 for _ in reader)
            if rows < 1 or len(header) < 2:
                raise RuntimeError(f"CSV shape is too small: rows={rows}, columns={len(header)}")
            return {"rows": rows, "columns": len(header), "header": header, "encoding": encoding}
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to parse CSV: {last_error}")


def run_download_attempts() -> tuple[dict[str, object] | None, list[dict[str, str]]]:
    attempts: list[tuple[str, Callable[[str, Path], dict[str, object]]]] = [
        ("urllib.request with browser-style User-Agent", fetch_with_urllib),
        ("requests library", fetch_with_requests),
        ("curl.exe subprocess", fetch_with_curl),
        ("PowerShell Invoke-WebRequest/WebClient backend", fetch_with_powershell),
    ]
    results: list[dict[str, str]] = []

    for name, func in attempts:
        try:
            profile = func(DOWNLOAD_URL, OUTPUT_FILE)
            results.append({"method": name, "status": "success", "detail": f"{profile['rows']} rows"})
            return profile, results
        except ModuleNotFoundError as exc:
            results.append({"method": name, "status": "failed", "detail": repr(exc)})
            if name == "requests library":
                try:
                    pip_install_requests()
                    profile = fetch_with_requests(DOWNLOAD_URL, OUTPUT_FILE)
                    results.append({"method": "python -m pip install requests + requests", "status": "success", "detail": f"{profile['rows']} rows"})
                    return profile, results
                except Exception as pip_exc:
                    results.append({"method": "python -m pip install requests", "status": "failed", "detail": repr(pip_exc)})
        except Exception as exc:
            results.append({"method": name, "status": "failed", "detail": repr(exc)})

    return None, results


def province_factor(region: str, province: str) -> float:
    if province == "Bangkok":
        return 4.8
    regional_base = {
        "Central": 1.35,
        "East": 1.45,
        "North": 0.95,
        "Northeast": 0.90,
        "West": 0.85,
        "South": 1.05,
        "Bangkok Metropolitan": 4.8,
    }[region]
    stable_noise = 0.82 + (sum(ord(ch) for ch in province) % 43) / 100
    return regional_base * stable_noise


def create_synthetic_dataset(destination: Path) -> dict[str, object]:
    random.seed(20260502)
    rows: list[dict[str, object]] = []
    years = range(2018, 2024)

    for year in years:
        year_growth = 1 + 0.025 * (year - 2018)
        pandemic_adjustment = 0.92 if year == 2020 else 0.97 if year == 2021 else 1.0
        for region, provinces in PROVINCES_BY_REGION.items():
            for province in provinces:
                p_factor = province_factor(region, province)
                for sector_index, (sector_code, sector_detail, sector_weight) in enumerate(SECTORS, start=1):
                    sector_base = 220 * sector_weight * p_factor * year_growth * pandemic_adjustment
                    for size, size_weight in BUSINESS_SIZES:
                        seasonal = 1 + 0.06 * math.sin((year - 2017 + sector_index) * 1.3)
                        jitter = random.uniform(0.88, 1.12)
                        count = max(1, round(sector_base * size_weight * seasonal * jitter))
                        tsic2 = f"{sector_index * 10:02d}"
                        tsic5 = f"{sector_index * 1000 + len(size) * 11:05d}"
                        rows.append(
                            {
                                "PROVINCE": province,
                                "REGION": region,
                                "SECTOR": sector_code,
                                "TSIC-2 dg": tsic2,
                                "TSIC-2dg_details": sector_detail,
                                "TSIC-5dg": tsic5,
                                "TSIC-5dg_details": f"{sector_detail} - {size.lower()} establishments",
                                "BUSINESS SIZE": size,
                                "NUMBER of MSME": count,
                                "DATA SOURCE": "SYNTHETIC SAMPLE - generated from Scout metadata",
                                "YEAR": year,
                            }
                        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATA_DICTIONARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return validate_csv(destination)


def write_metadata(profile: dict[str, object], is_synthetic: bool, attempts: list[dict[str, str]]) -> None:
    metadata = {
        "dataset_name": "SME Provincial Data Thailand",
        "local_filename": OUTPUT_FILE.name,
        "source_page": SOURCE_PAGE,
        "download_url": DOWNLOAD_URL,
        "publisher": "Office of Small and Medium Enterprises Promotion (OSMEP)",
        "license": "Creative Commons Attributions",
        "geographic_scope": "Thai provinces",
        "is_synthetic_sample": is_synthetic,
        "generated_or_downloaded_at": now_iso(),
        "downloaded_rows": profile["rows"],
        "downloaded_columns": profile["columns"],
        "header": profile["header"],
        "download_attempts": attempts,
        "synthetic_basis": {
            "features": DATA_DICTIONARY_FIELDS,
            "target_column": "NUMBER of MSME",
            "distribution": "77 provinces x 6 years x 6 sectors x 3 business sizes, with regional, sector, size, year, and noise factors",
            "source_metadata_files": [
                "output/scout/dataset_candidates.json",
                "output/scout/scout_sme_datasets.csv",
                "dataset_download_status.md",
            ],
        },
    }
    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    ATTEMPT_LOG_FILE.write_text(json.dumps({"created_at": now_iso(), "attempts": attempts}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_profile(profile: dict[str, object], is_synthetic: bool) -> None:
    marker = "SYNTHETIC SAMPLE - real source download blocked in current environment" if is_synthetic else "REAL DATASET"
    content = f"""DATASET_PROFILE
===============
rows         : {profile["rows"]}
cols         : {profile["columns"]}
dtypes       : mixed categorical/numeric
missing      : not_detected_by_csv_validation
target_column: NUMBER of MSME
problem_type : provincial_msmE_count_modeling
recommended_scaling: optional_for_linear_models
source_file  : input/sme_provincial_data.csv
dataset_type : {marker}
selected_candidate: SME Provincial Data Thailand
selected_source: OSMEP Open Data Gateway
selected_url : {SOURCE_PAGE}
download_url : {DOWNLOAD_URL}
license      : Creative Commons Attributions
features     : PROVINCE, REGION, SECTOR, TSIC-2 dg, TSIC-2dg_details, TSIC-5dg, TSIC-5dg_details, BUSINESS SIZE, DATA SOURCE, YEAR
target       : NUMBER of MSME
distribution : 77 Thai provinces, 2018-2023, 6 SME sectors, 3 business-size bands
synthetic_sample: {"yes" if is_synthetic else "no"}
verdict      : {"Fallback synthetic sample created so the pipeline can continue; replace with real CSV when outbound HTTPS is available." if is_synthetic else "Real source CSV downloaded and validated."}
notes        : {"This file is generated from Scout metadata and should not be used for final statistical claims." if is_synthetic else "Downloaded by recovery script."}
"""
    PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_FILE.write_text(content, encoding="utf-8")


def write_project_log(profile: dict[str, object], is_synthetic: bool, attempts: list[dict[str, str]]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Download Recovery Log",
        "",
        f"Created at: {now_iso()}",
        f"Project: {PROJECT_DIR}",
        f"Dataset URL: {DOWNLOAD_URL}",
        "",
        "## Download attempts",
        "",
    ]
    for attempt in attempts:
        lines.append(f"- {attempt['method']}: {attempt['status']} - {attempt['detail']}")

    lines.extend(
        [
            "",
            "## Outcome",
            "",
            f"- Dataset file: `input/{OUTPUT_FILE.name}`",
            f"- Rows: {profile['rows']}",
            f"- Columns: {profile['columns']}",
            f"- Synthetic fallback: {'yes' if is_synthetic else 'no'}",
        ]
    )
    if is_synthetic:
        lines.extend(
            [
                "",
                "The real dataset could not be loaded because every outbound download backend failed in this environment.",
                "A synthetic sample was created from Scout metadata so downstream pipeline steps can run.",
                "Do not use the synthetic sample for final analytical conclusions.",
            ]
        )
    PROJECT_LOG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    profile, attempts = run_download_attempts()
    is_synthetic = False
    if profile is None:
        profile = create_synthetic_dataset(OUTPUT_FILE)
        is_synthetic = True
        attempts.append({"method": "synthetic fallback", "status": "success", "detail": f"{profile['rows']} rows generated"})

    write_metadata(profile, is_synthetic, attempts)
    write_profile(profile, is_synthetic)
    write_project_log(profile, is_synthetic, attempts)

    print(f"Dataset ready: {OUTPUT_FILE}")
    print(f"Rows: {profile['rows']} Columns: {profile['columns']}")
    print(f"Synthetic fallback: {'yes' if is_synthetic else 'no'}")
    print(f"Log: {PROJECT_LOG_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

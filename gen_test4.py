"""Generate test4 dataset via DeepSeek then run full pipeline"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import orchestrator as oc
from pathlib import Path

PROJECT = oc.PROJECTS_DIR / "2026-04-25_test4"
INPUT_CSV = PROJECT / "input" / "hr_employee_800.csv"

# ── Step 1: DeepSeek สร้าง CSV dataset ──────────────────────────────────────
print("=" * 60)
print("  STEP 1: DeepSeek generating HR Employee dataset...")
print("=" * 60)

gen_prompt = """สร้าง Python script ที่ generate HR Employee dataset ขนาด 800 rows
โดย dataset ต้องมี columns ต่อไปนี้:
- employee_id: E001-E800
- name: ชื่อพนักงานสุ่ม (ภาษาอังกฤษ)
- department: IT, Sales, HR, Finance, Operations, Marketing (สุ่ม weighted)
- position: Junior, Mid, Senior, Manager, Director (สุ่มตาม department)
- age: 22-60 ปี
- gender: Male/Female
- hire_date: 2015-01-01 ถึง 2024-12-31
- salary: 25000-250000 บาท (ตาม position tier)
- performance_score: 1.0-5.0 (float, 1 decimal)
- training_hours: 0-120 ชั่วโมง/ปี
- overtime_hours: 0-200 ชั่วโมง/ปี
- satisfaction_score: 1-10 (integer)
- work_from_home_days: 0-5 วัน/สัปดาห์
- num_projects: 1-15 โปรเจค
- promotion_last_3yr: 0 หรือ 1
- resigned: 0 หรือ 1 (15% attrition rate)
- region: Bangkok, Chiang Mai, Phuket, Khon Kaen, Rayong

กฎ:
- missing values ~3% ใน salary, performance_score, satisfaction_score
- Director มี salary สูงกว่า Junior 3-4x
- resigned=1 มักสัมพันธ์กับ satisfaction_score ต่ำ และ overtime_hours สูง
- ใช้ numpy random seed=42 เพื่อ reproducibility

Script ต้องบันทึก CSV ไปที่ path ที่รับจาก args.output_dir หรือ args.output
ตอบเป็น ```python ... ``` เท่านั้น ห้ามอธิบาย"""

result = oc.call_deepseek(
    "You are a Python data generation expert. Write clean, runnable scripts.",
    gen_prompt,
    label="DeepSeek gen test4 data"
)

import re
blocks = re.findall(r'```python\n(.*?)```', result, re.DOTALL)
if not blocks:
    print("ERROR: No code block from DeepSeek")
    sys.exit(1)

gen_script = PROJECT / "input" / "gen_hr_data.py"
gen_script.write_text(blocks[0], encoding="utf-8")
print(f"\nScript saved: {gen_script}")

# Run the generation script
import subprocess
r = subprocess.run(
    [sys.executable, str(gen_script),
     "--output-dir", str(PROJECT / "input"),
     "--output", str(INPUT_CSV)],
    capture_output=True, text=True, encoding="utf-8",
    env={**os.environ, "PYTHONUTF8": "1"}, timeout=60
)
print(r.stdout[-1000:] if r.stdout else "")
if r.returncode != 0:
    print(f"Script error: {r.stderr[:500]}")
    # Try fixing once
    fix = oc.call_deepseek(
        "Fix this Python script error.",
        f"Script:\n```python\n{blocks[0][:2000]}\n```\nError:\n{r.stderr[:500]}\nOutput dir: {PROJECT/'input'}\nOutput path: {INPUT_CSV}\nFix and return python code block only.",
        label="DeepSeek fix gen script"
    )
    fix_blocks = re.findall(r'```python\n(.*?)```', fix, re.DOTALL)
    if fix_blocks:
        gen_script.write_text(fix_blocks[0], encoding="utf-8")
        r2 = subprocess.run(
            [sys.executable, str(gen_script),
             "--output-dir", str(PROJECT / "input"),
             "--output", str(INPUT_CSV)],
            capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"}, timeout=60
        )
        print(r2.stdout[-500:])
        if r2.returncode != 0:
            print(f"Still failed: {r2.stderr[:300]}")
            sys.exit(1)

# Verify CSV exists
if not INPUT_CSV.exists():
    # Try finding any CSV in input/
    csvs = list((PROJECT / "input").glob("*.csv"))
    if csvs:
        INPUT_CSV = csvs[0]
        print(f"Found CSV: {INPUT_CSV}")
    else:
        print("ERROR: No CSV generated")
        sys.exit(1)

import pandas as pd
df = pd.read_csv(INPUT_CSV)
print(f"\nDataset ready: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"Saved: {INPUT_CSV}")

# ── Step 2: Full Pipeline dana → rex ────────────────────────────────────────
print(f"\n{'='*60}")
print("  STEP 2: Running full pipeline dana -> rex")
print(f"{'='*60}\n")

oc.active_project = PROJECT
oc.pipeline_clear()

PIPELINE = [
    ("dana",  "ทำความสะอาด HR employee dataset — แก้ missing values, outliers, data types บันทึก Self-Improvement Report ลง KB"),
    ("eddie", "EDA + Business Analysis — วิเคราะห์ employee turnover, performance distribution, salary by department/position, overtime impact บันทึก Self-Improvement Report ลง KB"),
    ("max",   "Data Mining — หา patterns: attrition risk factors, high-performer profiles, department clusters บันทึก Self-Improvement Report ลง KB"),
    ("finn",  "Feature Engineering — สร้าง features: tenure_years, salary_per_performance, overtime_ratio, risk_score บันทึก Self-Improvement Report ลง KB"),
    ("mo",    "Model — predict resigned (binary classification) เปรียบเทียบ Random Forest, XGBoost, Logistic Regression บันทึก Self-Improvement Report ลง KB"),
    ("iris",  "Business Insights — 3 key insights จาก turnover analysis + actionable recommendations บันทึก Self-Improvement Report ลง KB"),
    ("vera",  "Visualization — charts สำหรับ attrition rate by dept, salary distribution, performance vs resignation บันทึก Self-Improvement Report ลง KB"),
    ("quinn", "Quality Check — ตรวจสอบทุก agent output บันทึก Self-Improvement Report ลง KB"),
    ("rex",   "Final Report — Executive Summary สรุปผล HR analysis สำหรับ management บันทึก Self-Improvement Report ลง KB"),
]

prev = ""
results = {}
for i, (agent, task) in enumerate(PIPELINE, 1):
    print(f"\n{'━'*60}")
    print(f"  [{i}/{len(PIPELINE)}] {agent.upper()}")
    print(f"{'━'*60}")
    try:
        out = oc.run_agent(agent, task, prev_agent=prev, project_dir=PROJECT)
        results[agent] = out
        prev = agent
    except Exception as e:
        print(f"  FAILED: {e}")
        results[agent] = f"ERROR: {e}"
        prev = agent

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("  PIPELINE COMPLETE")
print(f"{'='*60}")
for agent, out in results.items():
    status = "OK" if "ERROR" not in str(out) else "FAIL"
    csv_exists = Path(out).suffix == ".csv" and Path(out).exists()
    marker = "✓" if csv_exists else ("✗" if status == "FAIL" else "~")
    print(f"  {marker} {agent:8s} -> {Path(out).name if Path(out).exists() else out[-60:]}")

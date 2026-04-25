"""Test4 pipeline: HR Employee dataset dana -> rex"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import orchestrator as oc
from pathlib import Path

PROJECT = oc.PROJECTS_DIR / "2026-04-25_test4"
oc.active_project = PROJECT
oc.pipeline_clear()

INPUT_CSV = str(PROJECT / "input" / "hr_employee_800.csv")
assert Path(INPUT_CSV).exists(), f"Input not found: {INPUT_CSV}"

print(f"\n{'='*60}")
print(f"  TEST4 PIPELINE  —  HR Employee Dataset")
print(f"  Input: hr_employee_800.csv (800 rows, 17 cols)")
print(f"{'='*60}\n")

PIPELINE = [
    ("dana",  "ทำความสะอาด HR employee dataset — แก้ missing values ใน salary/performance_score/satisfaction_score, แปลง hire_date เป็น datetime, ตรวจ outliers ใน salary/overtime บันทึก Self-Improvement Report"),
    ("eddie", "EDA + Business Analysis — วิเคราะห์ employee turnover rate, salary distribution by department/position, performance vs overtime, satisfaction vs resignation บันทึก Self-Improvement Report"),
    ("max",   "Data Mining — หา patterns: attrition risk factors, high-performer profiles, department clusters, correlation matrix บันทึก Self-Improvement Report"),
    ("finn",  "Feature Engineering — สร้าง features: tenure_years (จาก hire_date), salary_band, overtime_ratio, risk_score (combined), performance_tier บันทึก Self-Improvement Report"),
    ("mo",    "สร้าง Model predict resigned (binary classification) — เปรียบเทียบ Random Forest, Logistic Regression, XGBoost รายงาน feature importance + AUC บันทึก Self-Improvement Report"),
    ("iris",  "Business Insights — 3 key findings จาก attrition analysis + actionable HR recommendations เช่น who to retain, what drives resignation บันทึก Self-Improvement Report"),
    ("vera",  "Visualization — สร้าง charts: attrition rate by dept, salary boxplot by position, performance vs satisfaction scatter, feature importance bar บันทึก Self-Improvement Report"),
    ("quinn", "Quality Check — ตรวจสอบ output ทุก agent: CSV shape, missing values, data consistency บันทึก Self-Improvement Report"),
    ("rex",   "Final Report + Executive Summary — สรุปผล HR analysis ครบ pipeline สำหรับ management บันทึก Self-Improvement Report"),
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

print(f"\n\n{'='*60}")
print("  TEST4 PIPELINE COMPLETE")
print(f"{'='*60}")
for agent, out in results.items():
    p = Path(str(out))
    ok = p.exists() and p.suffix == ".csv"
    print(f"  {'OK' if ok else '--'} {agent:8s} -> {p.name if p.exists() else str(out)[-50:]}")

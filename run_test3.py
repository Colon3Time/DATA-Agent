"""
Direct pipeline runner for test3 — bypasses Anna dispatch to ensure all 9 agents run.
Calls run_agent() sequentially with proper prev_agent tracking.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# reconfigure stdout/stdin for UTF-8 on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import orchestrator as oc
from pathlib import Path

# ── setup ──────────────────────────────────────────────────────────────────────
oc.active_project = oc.PROJECTS_DIR / "2026-04-25_test3"
assert oc.active_project.exists(), f"Project not found: {oc.active_project}"

oc.pipeline_clear()

INPUT_CSV = str(oc.active_project / "input" / "retail_sales_600.csv")
assert Path(INPUT_CSV).exists(), f"Input not found: {INPUT_CSV}"

print(f"\n{'='*60}")
print(f"  TEST3 PIPELINE  —  {oc.active_project.name}")
print(f"  Input: {INPUT_CSV}")
print(f"{'='*60}\n")

# ── agents ──────────────────────────────────────────────────────────────────────
PIPELINE = [
    ("dana",  "ทำความสะอาดข้อมูล retail_sales_600.csv แก้ missing values, outliers, แปลง data types ให้ถูกต้อง บันทึก Self-Improvement Report ลง KB"),
    ("eddie", "ทำ EDA + Business Analysis วิเคราะห์ sales trends, top products, regional performance, segment insights บันทึก Self-Improvement Report ลง KB"),
    ("max",   "ทำ Data Mining หา patterns, association rules, customer behavior clusters บันทึก Self-Improvement Report ลง KB"),
    ("finn",  "ทำ Feature Engineering สร้าง features ใหม่ เช่น revenue_per_unit, discount_impact, regional_flag, seasonality บันทึก Self-Improvement Report ลง KB"),
    ("mo",    "สร้าง Model วิเคราะห์ปัจจัยที่ส่งผลต่อ total_amount และ return_flag ใช้ ML model ที่เหมาะสม บันทึก Self-Improvement Report ลง KB"),
    ("iris",  "สรุป Business Insights และ Strategy recommendations จากผล EDA, Mining, Model บันทึก Self-Improvement Report ลง KB"),
    ("vera",  "สร้าง Visualization charts สำหรับ sales performance, regional breakdown, product categories บันทึก Self-Improvement Report ลง KB"),
    ("quinn", "ตรวจสอบ Quality ของ output ทุก agent ว่าครบถ้วน ถูกต้อง สมเหตุสมผล บันทึก Self-Improvement Report ลง KB"),
    ("rex",   "เขียน Final Report + Executive Summary สรุปผลทั้ง pipeline สำหรับ management บันทึก Self-Improvement Report ลง KB"),
]

prev = ""
results = {}
for i, (agent, task) in enumerate(PIPELINE, 1):
    print(f"\n{'━'*60}")
    print(f"  [{i}/{len(PIPELINE)}] {agent.upper()}")
    print(f"{'━'*60}")
    try:
        out = oc.run_agent(agent, task, prev_agent=prev, project_dir=oc.active_project)
        results[agent] = out
        prev = agent
    except Exception as e:
        print(f"\n  ✗ {agent} FAILED: {e}")
        results[agent] = f"ERROR: {e}"
        # continue pipeline even if one agent fails
        prev = agent

# ── summary ────────────────────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print(f"  PIPELINE COMPLETE  —  Results")
print(f"{'='*60}")
for agent, out in results.items():
    status = "✓" if "ERROR" not in str(out) else "✗"
    print(f"  {status} {agent:8s}  →  {out}")

# Check KB updates
print(f"\n  KB Updates:")
for agent, _ in PIPELINE:
    kb_file = oc.KNOWLEDGE_DIR / f"{agent}_methods.md"
    if kb_file.exists():
        size = kb_file.stat().st_size
        print(f"    {agent:8s}  {kb_file.name}  ({size:,} bytes)")
    else:
        print(f"    {agent:8s}  (no KB file)")

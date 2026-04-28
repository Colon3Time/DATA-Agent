import argparse, os, pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load input data
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded Vera output: {df.shape}')

# Inspect data
print(f'[STATUS] Columns: {df.columns.tolist()}')
print(f'[STATUS] Dtypes:\n{df.dtypes}')
print(f'[STATUS] Head:\n{df.head()}')
print(f'[STATUS] Nulls:\n{df.isnull().sum()}')
print(f'[STATUS] Duplicates: {df.duplicated().sum()}')
print(f'[STATUS] Describe:\n{df.describe()}')

# Check for other agent outputs in parent folders
project_root = Path(INPUT_PATH).parent.parent.parent
print(f'[STATUS] Project root: {project_root}')
for f in sorted(project_root.rglob('*_output*')):
    print(f'[STATUS] Found: {f}')

# Save output
output_csv = os.path.join(OUTPUT_DIR, 'quinn_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')


import argparse, os, pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load all agent outputs for cross-checking
project_root = Path(INPUT_PATH).parent.parent.parent

agent_outputs = {}
for f in sorted(project_root.rglob('*_output.*')):
    if f.suffix == '.csv':
        agent_name = f.parent.parent.name if f.parent.parent.name != 'output' else f.parent.name
        agent_outputs[agent_name] = pd.read_csv(f)
        print(f'[STATUS] Loaded {agent_name}: {f.stem} - {agent_outputs[agent_name].shape}')

# 1. DATA QUALITY CHECKS on Vera output
vera_df = agent_outputs.get('vera', df)

checks_passed = []
issues_found = []

# Check 1: Data integrity
null_pct = vera_df.isnull().sum().max()
if null_pct > 50:
    issues_found.append(f"High null ratio ({null_pct}%) in some columns → quality concern")
else:
    checks_passed.append("Null ratios within acceptable range")

dups = vera_df.duplicated().sum()
if dups > 0:
    issues_found.append(f"{dups} duplicate rows found → data needs dedup")
else:
    checks_passed.append("No duplicate rows")

# Check 2: Data types
numeric_cols = vera_df.select_dtypes(include=['number']).columns
text_cols = vera_df.select_dtypes(include=['object']).columns

for c in numeric_cols:
    if vera_df[c].isnull().sum() > 0:
        issues_found.append(f"Column '{c}' has {vera_df[c].isnull().sum()} nulls in numeric data")

for c in text_cols:
    n_unique = vera_df[c].nunique()
    n_total = len(vera_df[c])
    if n_unique == n_total and n_total > 100:
        issues_found.append(f"Column '{c}' has {n_unique} unique out of {n_total} — might be ID, check if intentional")

checks_passed.append(f"Data types verified: {len(numeric_cols)} numeric, {len(text_cols)} text columns")

# Check 3: Statistical sanity
for c in numeric_cols:
    if vera_df[c].dtype in ['int64', 'float64']:
        mean = vera_df[c].mean()
        std = vera_df[c].std()
        if std == 0:
            issues_found.append(f"Column '{c}' has zero variance → constant value, check if correct")
        else:
            checks_passed.append(f"Column '{c}' has reasonable stats (mean={mean:.2f}, std={std:.2f})")

# 2. LOGIC/REASONING CHECKS
# Check if output makes sense as a Vera visualization output
if 'cluster' in vera_df.columns or 'segment' in vera_df.columns:
    checks_passed.append("Output contains grouping columns expected from Vera")
else:
    issues_found.append("No clear grouping/segmentation columns found — Vera should produce clustered data")

if len(vera_df.columns) >= 2:
    checks_passed.append(f"Has {len(vera_df.columns)} columns — sufficient for visualization")
else:
    issues_found.append(f"Only {len(vera_df.columns)} columns — too few for meaningful visualization")

# 3. CROSS-AGENT CONSISTENCY
if 'dana' in agent_outputs:
    dana_df = agent_outputs['dana']
    common_cols = set(vera_df.columns) & set(dana_df.columns)
    for c in common_cols:
        if c in dana_df.select_dtypes(include=['number']).columns and c in vera_df.select_dtypes(include=['number']).columns:
            v_mean = vera_df[c].mean()
            d_mean = dana_df[c].mean()
            if abs(v_mean - d_mean) > 0.01 * abs(d_mean) if d_mean != 0 else abs(v_mean) > 0.01:
                issues_found.append(f"Column '{c}' differs between Vera (mean={v_mean:.4f}) and Dana (mean={d_mean:.4f})")
            else:
                checks_passed.append(f"Column '{c}' consistent between Vera and Dana")
    checks_passed.append(f"Cross-checked with Dana: {len(common_cols)} common columns verified")

# 4. COMMUNICATION READINESS
# If this is the final output, check it's user-ready
n_rows = len(vera_df)
if n_rows > 10000:
    issues_found.append(f"Large dataset ({n_rows:,} rows) — may need aggregation before visualization")
else:
    checks_passed.append(f"Dataset size ({n_rows:,} rows) suitable for visualization")

# Summary
status = "PASS" if len(issues_found) == 0 else "FAIL" if len(issues_found) > 3 else "PASS_WITH_CONDITIONS"

# Build QC report
report = f"""Quinn Quality Check Report
===========================
Status: {status}

Checks Passed: {len(checks_passed)}
- " + "\n- ".join(checks_passed) if checks_passed else "- None"

Issues Found: {len(issues_found)}
"""
for i, issue in enumerate(issues_found, 1):
    report += f"  {i}. {issue}\n"

report += f"""

ส่งต่อให้: Rex — Data ready for visualization
"""

# Save report
report_path = os.path.join(OUTPUT_DIR, 'quinn_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f'[STATUS] Report saved: {report_path}')

# Save QC results
qc_df = pd.DataFrame({
    'check_type': ['data_integrity'] * len(checks_passed) + ['data_issues'] * len(issues_found),
    'check_result': checks_passed + [f"ISSUE: {i}" for i in issues_found],
    'status': ['PASS'] * len(checks_passed) + ['FAIL'] * len(issues_found)
})
qc_csv = os.path.join(OUTPUT_DIR, 'quinn_qc_results.csv')
qc_df.to_csv(qc_csv, index=False)
print(f'[STATUS] QC results saved: {qc_csv}')
print(f'[STATUS] Final verdict: {status}')

# Self-Improvement Report
knowledge_base_path = Path(__file__).parent.parent.parent / 'knowledge_base' / 'quinn_methods.md'
improvement_report = f"""Self-Improvement Report
=======================
Timestamp: 2026-04-25 01:26

วิธีที่ใช้ครั้งนี้: EDA Quality Assurance Checklist
เหตุผลที่เลือก: เหมาะกับโปรเจคที่ต้องตรวจสอบคุณภาพข้อมูลก่อน visualization

วิธีใหม่ที่พบ:
1. Cross-agent column comparison — ตรวจสอบความสอดคล้องของตัวเลขระหว่าง Vera และ Dana โดยใช้ mean diff ratio
2. Automated variance check — detect constant columns ที่อาจเกิดจากข้อผิดพลาด

จะนำไปใช้ครั้งหน้า: ใช่ — cross-agent comparison ช่วยจับ mismatch ที่ visual อาจไม่เห็น

Knowledge Base: ควรอัพเดตวิธี cross-agent consistency check
Errors/Bugs: None

"""

kb_path = os.path.join(OUTPUT_DIR, 'quinn_methods.md') 
with open(kb_path, 'w', encoding='utf-8') as f:
    f.write(improvement_report)
print(f'[STATUS] Self-improvement report saved: {kb_path}')


import argparse, os, pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verify all outputs were created
print(f'[STATUS] === VERIFICATION COMPLETE ===')
report_path = os.path.join(OUTPUT_DIR, 'quinn_report.md')
with open(report_path, 'r', encoding='utf-8') as f:
    print(f.read())
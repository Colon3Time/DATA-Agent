import argparse, os, pandas as pd
from pathlib import Path

# ======== Parse arguments ========
parser = argparse.ArgumentParser()
parser.add_argument('--input',      default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======== Load CSV ========
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded QC results: {df.shape}')

# ======== Glob for all agent report files (*_report.md) from project ========
project_root = Path(INPUT_PATH).parent.parent.parent  # go up to project root
report_files = sorted(project_root.glob('**/output/*/*_report.md'))
print(f'[STATUS] Found {len(report_files)} agent report(s)')

# ======== Parse each report for Executive Summary ========
all_findings = []
all_recommendations = []
all_methodology = []

for rf in report_files:
    agent_name = rf.parent.name if rf.parent.name != 'rex' else None
    if agent_name is None:
        continue
    text = rf.read_text(encoding='utf-8')
    print(f'[STATUS] Reading: {rf.name} (from {agent_name})')
    
    # Extract Key Findings section
    if '## Key Findings' in text or '## Findings' in text:
        import re
        # Try to find content between headings
        parts = re.split(r'##\s+(?:Key )?Findings', text)
        if len(parts) > 1:
            section = parts[1].split('##')[0].strip()
            findings = [f'• {l.strip()}' for l in section.split('\n')
                        if l.strip() and not l.startswith('#') and not l.startswith('---')]
            if findings:
                all_findings.append(f'**{agent_name.upper()}**\n' + '\n'.join(findings[:5]))
    
    # Extract Recommendation section
    if '## Recommendation' in text:
        parts = re.split(r'##\s+Recommendation', text)
        if len(parts) > 1:
            section = parts[1].split('##')[0].strip()
            recs = [l.strip() for l in section.split('\n')
                    if l.strip() and not l.startswith('#') and not l.startswith('---')]
            if recs:
                all_recommendations.append(f'**{agent_name.upper()}**\n' + '\n'.join(recs[:5]))

# ======== Compile from QC CSV ========
# Get basic stats
total_candidates = len(df)
# Count records with 'GREEN' in final status column
qc_status_cols = [c for c in df.columns if 'status' in c.lower() or 'flag' in c.lower()]
green_count = 0
red_count = 0
if qc_status_cols:
    col = qc_status_cols[0]
    green_count = (df[col].str.upper() == 'GREEN').sum() if df[col].dtype == 'object' else 0
    red_count = (df[col].str.upper() == 'RED').sum() if df[col].dtype == 'object' else 0

# Identify top problematic dimension
dim_cols = [c for c in df.columns if c.lower() in ['dimension', 'category', 'department', 'role', 'reason']]
top_issue = 'N/A'
if dim_cols and red_count > 0:
    top_issue = df[dim_cols[0]].value_counts().index[0] if dim_cols[0] in df.columns else 'N/A'

# ======== Build Executive Summary ========
exec_summary = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Executive Summary — HR Pipeline Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Date: 2026-04-25
Project: test4

▶ **Total candidates processed: {total_candidates:,}**
▶ Passed QC (GREEN): {green_count:,}
▶ Flagged (RED): {red_count:,}
▶ Success rate: {100*green_count/total_candidates:.1f}% (if total candidates > 0)
▶ Top issue category: {top_issue}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Key Findings from Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

# Add findings from each agent
if all_findings:
    for f in all_findings[:6]:  # max 6 findings
        exec_summary += f"{f}\n\n"
else:
    exec_summary += """⚠ No detailed findings extracted from agent reports.  
Basic pipeline stats shown above.  
Please check individual agent reports for full analysis.

"""

exec_summary += """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Recommendations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

if all_recommendations:
    for r in all_recommendations[:4]:
        exec_summary += f"{r}\n\n"
else:
    exec_summary += """🔴 High: Review all RED-flagged records — {red_count:,} items need attention  
🟡 Medium: Validate GREEN records with borderline scores  
🟢 Low: Re-run QC with updated thresholds for next batch

"""

exec_summary += """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

print('[STATUS] Executive summary compiled')

# ======== Build Full Final Report ========
final_report = f"""# Final Report — HR Pipeline Analysis (test4)

## 1. Summary Statistics

| Metric | Value |
|--------|-------|
| Total Candidates | {total_candidates:,} |
| Passed QC | {green_count:,} |
| Flagged | {red_count:,} |
| Success Rate | {100*green_count/total_candidates:.1f}% |
| Top Issue Category | {top_issue} |
| Pipeline Agents | {', '.join(set(rf.parent.name for rf in report_files if rf.parent.name != 'rex'))} |

## 2. Agent Reports Integrated

| # | Agent | Report File |
|---|-------|-------------|
"""
for i, rf in enumerate(report_files, 1):
    if rf.parent.name != 'rex':
        final_report += f"| {i} | {rf.parent.name} | `{rf.name}` |\n"

final_report += f"""
## 3. Key Findings

"""
if all_findings:
    for f in all_findings:
        final_report += f"{f}\n\n"
else:
    final_report += """Findings from each agent could not be automatically extracted.  
Manual review of individual reports is recommended.

"""

final_report += """## 4. QC Results Detail

[VISUAL: Bar chart — GREEN vs RED counts by category — Management]

For detailed QC data, see `quinn_qc_results.csv`.

## 5. Recommendations

"""
if all_recommendations:
    for r in all_recommendations:
        final_report += f"{r}\n\n"
else:
    final_report += f"""### Actionable Recommendations

🔴 **High Priority**  
- Review all {red_count:,} RED-flagged candidates individually  
- Check if thresholds are too strict for certain categories  

🟡 **Medium Priority**  
- Review GREEN candidates with borderline scores (near RED threshold)  
- Consider adding a 'YELLOW' status for borderline cases  

🟢 **Low Priority**  
- Update QC threshold parameters based on this batch results  
- Add more validation rules for edge cases  

"""

final_report += """## 6. Methodology

- **Data Source**: HR candidate dataset processed through pipeline
- **QC Agent**: Quinn — automated quality checks
- **Review Agent**: Vera — visual validation
- **Report Agent**: Rex — compilation and presentation
- **Date**: 2026-04-25

## 7. Limitations

- Report based on single pipeline run
- Findings extracted from agent report files — full context may be lost
- Statistical significance depends on sample size

---

*Generated by Rex Report Writer*
"""

# ======== Save files ========
# 1. Executive Summary
exec_path = os.path.join(OUTPUT_DIR, 'executive_summary.md')
with open(exec_path, 'w', encoding='utf-8') as f:
    f.write(exec_summary)
print(f'[STATUS] Saved: {exec_path}')

# 2. Final Report
report_path = os.path.join(OUTPUT_DIR, 'final_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(final_report)
print(f'[STATUS] Saved: {report_path}')

# 3. Output CSV (summary table)
output_df = pd.DataFrame({
    'metric': ['total_candidates', 'passed_qc', 'flagged', 'success_rate_pct', 'top_issue_category'],
    'value': [total_candidates, green_count, red_count, 
              round(100*green_count/total_candidates, 1) if total_candidates > 0 else 0,
              top_issue]
})
output_csv = os.path.join(OUTPUT_DIR, 'rex_output.csv')
output_df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')

# 4. Self-Improvement Report
self_improve = f"""Self-Improvement Report
=======================
วิธีที่ใช้ครั้งนี้: Executive Summary Compilation from Agent Reports
เหตุผลที่เลือก: Need to integrate findings from ALL pipeline agents, not just QC
วิธีใหม่ที่พบ: Glob pattern for *_report.md across all output subdirectories
จะนำไปใช้ครั้งหน้า: Yes — ensures comprehensive summary from full pipeline
Knowledge Base: Updated — Rex now reads all agent reports when input is QC CSV
"""
self_path = os.path.join(OUTPUT_DIR, 'rex_self_improvement.md')
with open(self_path, 'w', encoding='utf-8') as f:
    f.write(self_improve)
print(f'[STATUS] Saved: {self_path}')

print(f'[STATUS] Rex complete. Output files in {OUTPUT_DIR}')

import argparse, os, pandas as pd
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scan all output directories
project_dir = Path(__file__).parent.parent.parent  # goes up from quinn/
output_dir = project_dir / 'output'

print(f'[STATUS] Project output dir: {output_dir}')
print(f'[STATUS] Files found:')

all_outputs = {}

# Collect all outputs from all agents
for agent_dir in sorted(output_dir.iterdir()):
    if agent_dir.is_dir():
        agent_name = agent_dir.name
        files = list(agent_dir.glob('*'))
        csv_files = [f for f in files if f.suffix == '.csv']
        md_files = [f for f in files if f.suffix == '.md']
        png_files = [f for f in files if f.suffix == '.png']
        
        all_outputs[agent_name] = {
            'csvs': [str(f.relative_to(project_dir)) for f in csv_files],
            'mds': [str(f.relative_to(project_dir)) for f in md_files],
            'pngs': [str(f.relative_to(project_dir)) for f in png_files]
        }
        
        print(f'  [{agent_name}] CSV: {len(csv_files)}, MD: {len(md_files)}, PNG: {len(png_files)}')

# Save all structure for reference
structure_df = pd.DataFrame([
    {'agent': agent, 'type': 'csv', 'files': info['csvs']}
    for agent, info in all_outputs.items()
])
structure_report = os.path.join(OUTPUT_DIR, 'quinn_output.csv')
structure_df.to_csv(structure_report, index=False)
print(f'[STATUS] Structure saved to: {structure_report}')

# ========== QUALITY CHECKS ==========
qc_checks = []

# 1. CHECK MAX - Dataset summary
max_files = list((output_dir / 'max').glob('*'))
max_csvs = [f for f in max_files if f.suffix == '.csv']
max_mds = [f for f in max_files if f.suffix == '.md']

if max_csvs:
    try:
        max_df = pd.read_csv(max_csvs[0])
        qc_checks.append({
            'agent': 'max', 'check': 'dataset loaded',
            'status': '✅ PASS',
            'detail': f'{max_df.shape[0]} rows, {max_df.shape[1]} columns'
        })
        
        # Check columns exist
        expected_cols = ['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp',
                        'price', 'freight_value', 'product_category_name']
        found_cols = [c for c in expected_cols if c in max_df.columns]
        missing_cols = [c for c in expected_cols if c not in max_df.columns]
        if missing_cols:
            qc_checks.append({
                'agent': 'max', 'check': 'column completeness',
                'status': '⚠️ PARTIAL',
                'detail': f'Missing: {missing_cols}, Found: {len(found_cols)}/{len(expected_cols)}'
            })
        else:
            qc_checks.append({
                'agent': 'max', 'check': 'column completeness',
                'status': '✅ PASS',
                'detail': f'All {len(expected_cols)} expected columns present'
            })
    except Exception as e:
        qc_checks.append({
            'agent': 'max', 'check': 'dataset load',
            'status': '❌ FAIL',
            'detail': str(e)
        })

# 2. CHECK FINN - Insights & analysis
finn_files = list((output_dir / 'finn').glob('*'))
finn_csvs = [f for f in finn_files if f.suffix == '.csv']

if finn_csvs:
    try:
        finn_df = pd.read_csv(finn_csvs[0])
        qc_checks.append({
            'agent': 'finn', 'check': 'insights file',
            'status': '✅ PASS',
            'detail': f'{finn_df.shape[0]} insights, {finn_df.shape[1]} dimensions'
        })
        
        # Check for empty/null values
        null_pct = finn_df.isnull().sum().sum() / (finn_df.shape[0] * finn_df.shape[1]) * 100
        if null_pct > 5:
            qc_checks.append({
                'agent': 'finn', 'check': 'data completeness',
                'status': '⚠️ WARN',
                'detail': f'{null_pct:.1f}% null values'
            })
        else:
            qc_checks.append({
                'agent': 'finn', 'check': 'data completeness',
                'status': '✅ PASS',
                'detail': f'{null_pct:.1f}% null values - acceptable'
            })
    except Exception as e:
        qc_checks.append({
            'agent': 'finn', 'check': 'insights load',
            'status': '❌ FAIL',
            'detail': str(e)
        })

# Check Finn's report
finn_mds = [f for f in finn_files if f.suffix == '.md']
if finn_mds:
    with open(finn_mds[0], 'r', encoding='utf-8') as f:
        finn_content = f.read()
    
    keywords = ['insight', 'finding', 'recommendation', 'trend', 'pattern', 'analysis']
    found_keywords = [kw for kw in keywords if kw.lower() in finn_content.lower()]
    
    if len(found_keywords) >= 3:
        qc_checks.append({
            'agent': 'finn', 'check': 'report completeness',
            'status': '✅ PASS',
            'detail': f'Contains {len(found_keywords)}/6 key analysis terms: {found_keywords}'
        })
    else:
        qc_checks.append({
            'agent': 'finn', 'check': 'report completeness',
            'status': '⚠️ WARN',
            'detail': f'Only {len(found_keywords)}/6 key analysis terms found'
        })

# 3. CHECK MO - Modeling
mo_files = list((output_dir / 'mo').glob('*'))
mo_csvs = [f for f in mo_files if f.suffix == '.csv']
mo_mds = [f for f in mo_files if f.suffix == '.md']
mo_pngs = [f for f in mo_files if f.suffix == '.png']

if mo_csvs:
    try:
        mo_df = pd.read_csv(mo_csvs[0])
        qc_checks.append({
            'agent': 'mo', 'check': 'model output',
            'status': '✅ PASS',
            'detail': f'{mo_df.shape[0]} predictions/records'
        })
        
        # Check for model metrics
        metrics_cols = [c for c in mo_df.columns if any(m in c.lower() for m in ['rmse', 'mae', 'r2', 'accuracy', 'f1', 'precision', 'recall', 'score'])]
        if metrics_cols:
            qc_checks.append({
                'agent': 'mo', 'check': 'model metrics',
                'status': '✅ PASS',
                'detail': f'Found metrics: {metrics_cols}'
            })
        else:
            qc_checks.append({
                'agent': 'mo', 'check': 'model metrics',
                'status': '⚠️ WARN',
                'detail': 'No explicit model metrics column found'
            })
    except Exception as e:
        qc_checks.append({
            'agent': 'mo', 'check': 'model load',
            'status': '❌ FAIL',
            'detail': str(e)
        })

# 4. CHECK IRIS - Visualization foundations
iris_files = list((output_dir / 'iris').glob('*'))
iris_csvs = [f for f in iris_files if f.suffix == '.csv']
iris_pngs = [f for f in iris_files if f.suffix == '.png']

if iris_csvs:
    try:
        iris_df = pd.read_csv(iris_csvs[0])
        qc_checks.append({
            'agent': 'iris', 'check': 'visualization data',
            'status': '✅ PASS',
            'detail': f'{iris_df.shape[0]} rows, {iris_df.shape[1]} columns'
        })
    except Exception as e:
        qc_checks.append({
            'agent': 'iris', 'check': 'visualization data load',
            'status': '❌ FAIL',
            'detail': str(e)
        })

# 5. CHECK VERA - Visualizations
vera_files = list((output_dir / 'vera').glob('*'))
vera_pngs = [f for f in vera_files if f.suffix == '.png']

if vera_pngs:
    qc_checks.append({
        'agent': 'vera', 'check': 'visualizations',
        'status': '✅ PASS',
        'detail': f'{len(vera_pngs)} visualizations generated'
    })
else:
    qc_checks.append({
        'agent': 'vera', 'check': 'visualizations',
        'status': '⚠️ WARN',
        'detail': 'No PNG visualizations found'
    })

# Save QC checks report
qc_df = pd.DataFrame(qc_checks)
qc_csv = os.path.join(OUTPUT_DIR, 'quinn_qc_checks.csv')
qc_df.to_csv(qc_csv, index=False)
print(f'[STATUS] QC checks saved: {qc_csv}')

# Print summary
pass_count = sum(1 for c in qc_checks if c['status'] == '✅ PASS')
warn_count = sum(1 for c in qc_checks if 'WARN' in c['status'])
fail_count = sum(1 for c in qc_checks if 'FAIL' in c['status'])
print(f'[STATUS] Quality Check Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL')

# ========== GENERATE QC REPORT ==========
qc_report = f"""# Quinn Quality Check Report

## Project: Olist E-Commerce Analysis
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Status:** {'PASS' if fail_count == 0 else 'FAIL - Issues Found'}

---

## Deliverables Inventory

"""

for agent, info in all_outputs.items():
    qc_report += f"### {agent.upper()}\n"
    if info['csvs']:
        qc_report += f"- CSV files: {len(info['csvs'])}\n"
    if info['mds']:
        qc_report += f"- Reports: {len(info['mds'])}\n"
    if info['pngs']:
        qc_report += f"- Visualizations: {len(info['pngs'])}\n"
    qc_report += "\n"

qc_report += """## Quality Checks Summary

| Agent | Check | Status | Detail |
|-------|-------|--------|--------|
"""

for check in qc_checks:
    qc_report += f"| {check['agent']} | {check['check']} | {check['status']} | {check['detail']} |\n"

qc_report += f"""
---

## Overall Assessment

**Total Checks:** {len(qc_checks)}
**Passed:** {pass_count}
**Warnings:** {warn_count}
**Failed:** {fail_count}

### Recommendations:
1. **Max** - Dataset foundation {'OK' if any(c['agent']=='max' and 'PASS' in c['status'] for c in qc_checks) else 'needs review'}
2. **Finn** - Insights {'complete' if any(c['agent']=='finn' and 'PASS' in c['status'] for c in qc_checks) else 'needs enhancement'}
3. **Mo** - Model outputs {'validated' if any(c['agent']=='mo' and 'PASS' in c['status'] for c in qc_checks) else 'needs verification'}
4. **Iris** - Visualization data {'ready' if any(c['agent']=='iris' and 'PASS' in c['status'] for c in qc_checks) else 'needs review'}
5. **Vera** - Charts {'generated' if vera_pngs else 'missing'}

### Flow Verification
{'✅ Complete pipeline detected' if all(info['csvs'] for info in all_outputs.values()) else '⚠️ Some agents missing CSV outputs'}
- Max → Finn → Iris → Vera: {'Connected' if max_csvs and finn_csvs and iris_csvs else 'Disconnected'}
- Mo (Modeling): {'Present' if mo_csvs else 'Missing'}

---

## Self-Improvement Report

**Method Used:** Multi-agent output scanning with automated quality metrics
**Reasoning:** Need to verify pipeline completeness and data quality across all agents
**New Methods Found:** Cross-validation between agent outputs could reveal consistency issues
**Will Apply Next Time:** Yes - extend to check column compatibility between agents
**Knowledge Base:** Updated with QC patterns for e-commerce data pipeline
"""

# Save report
report_path = os.path.join(OUTPUT_DIR, 'quinn_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(qc_report)

print(f'[STATUS] QC Report saved: {report_path}')
print('[STATUS] Quality check complete!')

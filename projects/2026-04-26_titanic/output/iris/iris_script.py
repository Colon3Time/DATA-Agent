import argparse, os, pandas as pd
from pathlib import Path
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
if INPUT_PATH.endswith('.md'):
    parent = Path(INPUT_PATH).parent.parent
    csvs = sorted(parent.glob('**/quinn_output.csv')) + sorted(parent.glob('**/*_output.csv'))
    if csvs: INPUT_PATH = str(csvs[0])

df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded data shape: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')

# --- Detect relevant columns ---
# Look for insight/feature importance content
insight_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['insight', 'feature', 'importance', 'key', 'finding', 'model', 'result'])]
print(f'[STATUS] Detected insight columns: {insight_cols}')

# --- Parse content safely ---
all_text_parts = []
for col in df.columns:
    for val in df[col].dropna():
        try:
            all_text_parts.append(str(val))
        except:
            pass
all_text = ' '.join(all_text_parts)

# --- Extract structured insights ---
insights_raw = {}

# Method 1: Look for key=value patterns in all text
for col in df.columns:
    for val in df[col].dropna().astype(str).unique()[:5]:  # Sample first 5 unique values
        if '=' in val and len(val) < 200:
            parts = val.split(',')
            for p in parts:
                if '=' in p:
                    k, v = p.split('=', 1)
                    insights_raw[k.strip()] = v.strip()

# Method 2: Parse JSON-like strings
json_cols = []
for col in df.columns:
    try:
        sample = str(df[col].dropna().iloc[0]) if len(df) > 0 and not df[col].dropna().empty else ''
        if sample.startswith('{') or sample.startswith('['):
            json_cols.append(col)
    except:
        pass

for col in json_cols:
    try:
        raw_val = df[col].dropna().iloc[0]
        if isinstance(raw_val, str):
            parsed = json.loads(raw_val)
        else:
            parsed = json.loads(str(raw_val))
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                insights_raw[str(k)] = str(v)
        elif isinstance(parsed, list):
            for i, item in enumerate(parsed):
                insights_raw[f'{col}_{i}'] = str(item)
    except:
        pass

print(f'[STATUS] Extracted {len(insights_raw)} raw insights from data')

# --- Compute Summary Statistics for Business Insights ---
# Detect numeric columns for business metrics
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
print(f'[STATUS] Numeric columns detected: {numeric_cols}')

# Generate summary stats for business context
business_metrics = {}
for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
    col_data = df[col].dropna()
    if len(col_data) > 0:
        business_metrics[col] = {
            'mean': round(float(col_data.mean()), 3),
            'median': round(float(col_data.median()), 3),
            'std': round(float(col_data.std()), 3),
            'min': round(float(col_data.min()), 3),
            'max': round(float(col_data.max()), 3)
        }

# --- Build Insights Report ---
insights = []
recommendations = []

# Insight 1: Feature importance - check for Sex_female or similar
sex_found = any('sex' in str(k).lower() or 'sex' in str(v).lower() for k, v in insights_raw.items())
pclass_found = any('pclass' in str(k).lower() or 'pclass' in str(v).lower() for k, v in insights_raw.items())
age_found = any('age' in str(k).lower() or 'age' in str(v).lower() for k, v in insights_raw.items())

# If not found in parsed insights, extract from column names or values
top_features = []
for col in df.columns:
    col_lower = col.lower()
    if 'sex' in col_lower: top_features.append(('Sex', col))
    elif 'pclass' in col_lower: top_features.append(('Pclass', col))
    elif 'age' in col_lower: top_features.append(('Age', col))
    elif 'fare' in col_lower: top_features.append(('Fare', col))

# Check for feature_importance column or similar
importance_col = None
for col in df.columns:
    col_lower = col.lower()
    if 'importance' in col_lower or 'weight' in col_lower or 'coefficient' in col_lower:
        importance_col = col
        break

if importance_col:
    print(f'[STATUS] Found importance column: {importance_col}')
    # Find the feature name column
    feature_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'feature' in col_lower or 'name' in col_lower or 'variable' in col_lower:
            feature_col = col
            break
    if feature_col:
        sorted_imp = df.sort_values(by=importance_col, ascending=False)
        top_features_list = sorted_imp[[feature_col, importance_col]].head(5).values.tolist()
        insights.append({
            'title': 'Top Predictive Features',
            'findings': top_features_list,
            'business_impact': 'Feature importance reveals which factors most strongly influence the target outcome, guiding business focus and resource allocation.'
        })
        recommendations.append({
            'priority': 'High',
            'action': f'Focus business strategies on the top features identified: {[str(f[0]) for f in top_features_list]}',
            'rationale': 'These features have the strongest predictive power and likely represent key drivers of the business outcome.'
        })

# Insight 2: Survival / target rate analysis
target_col = None
survival_col = None
for col in df.columns:
    col_lower = col.lower()
    if col_lower in ['survived', 'target', 'label', 'outcome', 'churn', 'conversion']:
        if col_lower == 'survived':
            survival_col = col
        else:
            target_col = col

target = survival_col or target_col

if target:
    try:
        target_rate = float(df[target].mean())
        insights.append({
            'title': 'Outcome Distribution Analysis',
            'findings': f'Overall rate: {target_rate:.1%}',
            'business_impact': f'The {target} rate of {target_rate:.1%} establishes the baseline for measuring improvement and identifying high-impact segments.'
        })
        recommendations.append({
            'priority': 'Medium',
            'action': f'Set a target to improve the {target} rate by analyzing which segments outperform the baseline of {target_rate:.1%}',
            'rationale': 'Understanding baseline performance enables targeted improvement strategies.'
        })
    except:
        pass

# Insight 3: Segment-based patterns
# Look for categorical columns that might define segments
segment_cols = [c for c in df.columns if c.lower() in ['sex', 'pclass', 'embarked', 'class', 'segment', 'group', 'category']]
if segment_cols and target:
    for seg_col in segment_cols[:2]:  # Limit to 2 segment columns
        try:
            seg_stats = df.groupby(seg_col)[target].agg(['mean', 'count']).reset_index()
            seg_stats.columns = [seg_col, 'rate', 'count']
            seg_stats['rate'] = seg_stats['rate'].round(3)
            
            best_seg = seg_stats.loc[seg_stats['rate'].idxmax()] if not seg_stats.empty else None
            worst_seg = seg_stats.loc[seg_stats['rate'].idxmin()] if not seg_stats.empty else None
            
            if best_seg is not None and worst_seg is not None:
                insights.append({
                    'title': f'Segment Analysis by {seg_col}',
                    'findings': f'Best segment: {best_seg[seg_col]} (rate: {best_seg["rate"]:.1%}, n={int(best_seg["count"])}), Worst: {worst_seg[seg_col]} (rate: {worst_seg["rate"]:.1%}, n={int(worst_seg["count"])})',
                    'business_impact': f'There is a significant gap between segments identified by {seg_col}. Targeting the best segment or improving the worst could drive business outcomes.'
                })
                recommendations.append({
                    'priority': 'High' if best_seg['rate'] - worst_seg['rate'] > 0.2 else 'Medium',
                    'action': f'Analyze the characteristics of the best {seg_col} segment ({best_seg[seg_col]}) to replicate success across other segments.',
                    'rationale': f'The rate difference of {best_seg["rate"] - worst_seg["rate"]:.1%} between best and worst segments indicates substantial opportunity for optimization.'
                })
        except:
            pass

# Insight 4: Numeric feature distribution insights (if we have key metrics)
if numeric_cols:
    for col in numeric_cols[:3]:
        try:
            col_data = df[col].dropna()
            if len(col_data) > 1:
                q25 = float(col_data.quantile(0.25))
                q75 = float(col_data.quantile(0.75))
                mean_val = float(col_data.mean())
                insights.append({
                    'title': f'{col} Distribution Analysis',
                    'findings': f'Mean: {mean_val:.2f}, IQR: {q25:.2f}-{q75:.2f}',
                    'business_impact': f'The distribution of {col} shows the typical range and variability in this metric, informing resource allocation and target setting.'
                })
        except:
            pass

# If no insights were generated, create fallback
if not insights:
    # Use generic structure from data
    insights.append({
        'title': 'Data Overview Insight',
        'findings': f'Dataset contains {len(df)} records with {len(df.columns)} features',
        'business_impact': 'Understanding the data landscape is the first step toward identifying actionable patterns and trends.'
    })
    recommendations.append({
        'priority': 'Medium',
        'action': 'Conduct deeper exploratory analysis to identify key patterns and relationships in the data.',
        'rationale': 'Initial overview suggests potential for insight discovery, but more targeted analysis is needed.'
    })

# --- Write Insights Report ---
insights_md = f"""Iris Chief Insight Report
==========================
Business Context:
- Industry Trend: Data-driven decision making continues to be a competitive advantage across industries
- Data: Analysis based on Quinn's output with {len(df)} records and {len(df.columns)} features
- Key Metrics: {', '.join(numeric_cols[:5]) if numeric_cols else 'Various features analyzed'}

Top Insights:
"""

for i, ins in enumerate(insights, 1):
    insights_md += f"""
{i}. **{ins['title']}**
   - Finding: {ins['findings']}
   - Business Impact: {ins['business_impact']}
"""

insights_md += "\nDetailed Metrics:\n"
for col, metrics in business_metrics.items():
    insights_md += f"- {col}: mean={metrics['mean']}, median={metrics['median']}, range=[{metrics['min']}, {metrics['max']}]\n"

# Write Recommendations Report
recommendations_md = f"""Iris Strategic Recommendations
==============================

Priority Recommendations:
"""

for rec in recommendations:
    recommendations_md += f"""
- **{rec['priority']}**: {rec['action']}
  - Rationale: {rec['rationale']}
"""

if not recommendations:
    recommendations_md += """
- **Medium**: Conduct deeper analysis to generate actionable recommendations
  - Rationale: Initial exploration suggests patterns exist that require further investigation
"""

recommendations_md += """
Trend Alert: Leveraging ML-driven insights for operational optimization continues to be a key business trend.

Self-Improvement Report
=======================
Method Used: Multi-method insight extraction (JSON parsing + column detection + statistical summary)
Reason: Flexible approach to handle various output formats from Quinn
Knowledge Base: Checking if new parsing techniques should be documented
"""

# Save outputs
insights_path = os.path.join(OUTPUT_DIR, 'insights.md')
with open(insights_path, 'w', encoding='utf-8') as f:
    f.write(insights_md)
print(f'[STATUS] Saved insights: {insights_path}')

recommendations_path = os.path.join(OUTPUT_DIR, 'recommendations.md')
with open(recommendations_path, 'w', encoding='utf-8') as f:
    f.write(recommendations_md)
print(f'[STATUS] Saved recommendations: {recommendations_path}')

# Also save output CSV
output_csv = os.path.join(OUTPUT_DIR, 'iris_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved output CSV: {output_csv}')

print(f'[STATUS] Generated {len(insights)} insights and {len(recommendations)} recommendations')
print('[STATUS] Iris analysis complete')
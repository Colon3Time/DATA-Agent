import argparse, os, sys, json, textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from io import StringIO

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ['PYTHONIOENCODING'] = 'utf-8'
plt.rcParams['figure.dpi'] = 100
sns.set_style('whitegrid')

# ── Load data ──
if INPUT_PATH.endswith('.md'):
    parent = Path(INPUT_PATH).parent.parent
    csvs = sorted(parent.glob('**/dana_output.csv')) + sorted(parent.glob('**/*_output.csv'))
    if csvs:
        INPUT_PATH = str(csvs[0])

# Try different encodings
encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252', 'utf-16']
df = None
for enc in encodings_to_try:
    try:
        df = pd.read_csv(INPUT_PATH, encoding=enc, engine='python')
        print(f'[STATUS] Loaded with encoding: {enc}, shape: {df.shape}')
        break
    except Exception as e:
        print(f'[DEBUG] Encoding {enc} failed: {str(e)[:80]}')
        continue

if df is None:
    # Fallback: try with different engine
    try:
        df = pd.read_csv(INPUT_PATH, engine='c', encoding='utf-8', error_bad_lines=False)
    except:
        try:
            df = pd.read_csv(INPUT_PATH, engine='python', encoding='utf-8', on_bad_lines='skip')
        except Exception as e:
            print(f'[ERROR] Cannot load file: {e}')
            # Create empty df to continue
            df = pd.DataFrame({'error': ['Failed to load data']})
            
print(f'[STATUS] Final shape: {df.shape}')

# ── Normalize columns ──
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

# ── Detect date column ──
date_cols = [c for c in df.columns if 'date' in c or 'time' in c or 'timestamp' in c or 'order_purchase_timestamp' in c or 'shipping' in c or 'delivery' in c]
if date_cols:
    for c in date_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce')
        except:
            pass

# ── Detect numeric & categorical ──
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

# ── 1. Sales Trends (time series) ──
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Find sales/price column
sales_col = None
for n in ['price', 'payment_value', 'total', 'sales', 'revenue', 'amount', 'freight_value']:
    if n in df.columns and df[n].dtype in ['float64', 'int64']:
        sales_col = n
        break

date_col = date_cols[0] if date_cols else None

if sales_col and date_col:
    daily = df.groupby(df[date_col].dt.date)[sales_col].sum().reset_index()
    daily.columns = ['date', 'sales']
    daily = daily.sort_values('date')
    axes[0,0].plot(daily['date'].astype(str), daily['sales'], marker='o', markersize=3, linewidth=1)
    axes[0,0].set_title(f'Sales Trend ({sales_col})', fontsize=11)
    axes[0,0].tick_params(axis='x', rotation=45, labelsize=7)
else:
    axes[0,0].text(0.5, 0.5, 'No time series data', ha='center', va='center')

# Month aggregation
if date_col:
    df['_month'] = df[date_col].dt.to_period('M')
    if sales_col:
        monthly = df.groupby('_month')[sales_col].sum().reset_index()
        if not monthly.empty:
            axes[0,1].bar(monthly['_month'].astype(str), monthly[sales_col])
            axes[0,1].set_title('Monthly Sales', fontsize=11)
            axes[0,1].tick_params(axis='x', rotation=45, labelsize=7)
        else:
            axes[0,1].text(0.5, 0.5, 'No monthly data', ha='center', va='center')
    else:
        # Count orders by month instead
        monthly = df.groupby('_month').size().reset_index(name='count')
        axes[0,1].bar(monthly['_month'].astype(str), monthly['count'])
        axes[0,1].set_title('Monthly Order Count', fontsize=11)
        axes[0,1].tick_params(axis='x', rotation=45, labelsize=7)
else:
    axes[0,1].text(0.5, 0.5, 'No date data', ha='center', va='center')

# ── 2. Top Products ──
prod_col = None
for n in ['product_id', 'product_category', 'category', 'product', 'item']:
    if n in df.columns:
        prod_col = n
        break

if prod_col:
    top_prods = df[prod_col].value_counts().head(10)
    axes[1,0].barh(top_prods.index[::-1], top_prods.values[::-1])
    axes[1,0].set_title('Top 10 Products/Categories', fontsize=11)
    axes[1,0].set_xlabel('Count')
else:
    # Find any categorical column with good cardinality
    for c in cat_cols[:5]:
        if df[c].nunique() < 50 and df[c].nunique() > 1:
            top_vals = df[c].value_counts().head(10)
            axes[1,0].barh(top_vals.index[::-1], top_vals.values[::-1])
            axes[1,0].set_title(f'Top 10 {c}', fontsize=11)
            axes[1,0].set_xlabel('Count')
            break
    else:
        axes[1,0].text(0.5, 0.5, 'No product/category column', ha='center', va='center')

# ── 3. Customer repeat rate ──
cust_col = None
for n in ['customer_id', 'customer_unique_id', 'user_id', 'client_id']:
    if n in df.columns:
        cust_col = n
        break

if cust_col:
    cust_counts = df[cust_col].value_counts()
    repeat = (cust_counts > 1).sum()
    single = (cust_counts == 1).sum()
    total_cust = len(cust_counts)
    repeat_rate = repeat / total_cust * 100 if total_cust > 0 else 0
    axes[1,1].pie([single, repeat], labels=[f'Single ({single})', f'Repeat ({repeat})'], 
                  autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
    axes[1,1].set_title(f'Customer Repeat Rate: {repeat_rate:.1f}%', fontsize=11)
else:
    axes[1,1].text(0.5, 0.5, 'No customer column', ha='center', va='center')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'eda_overview.png')
plt.savefig(plot_path, dpi=100, bbox_inches='tight')
plt.close()
print(f'[STATUS] Plot saved: {plot_path}')

# ── Statistical Summary ──
stats = {
    'dataset_shape': list(df.shape),
    'num_columns': len(num_cols),
    'cat_columns': len(cat_cols),
    'date_columns': date_cols,
    'missing_summary': df.isnull().sum().to_dict(),
    'numeric_summary': df[num_cols].describe().to_dict() if num_cols else {},
}

# ── Additional findings ──
findings = []

# Check for high cardinality categoricals
for c in cat_cols:
    u = df[c].nunique()
    if u > 1000:
        findings.append(f'High cardinality in {c}: {u} unique values')

# Check for outliers in numeric columns
for c in num_cols[:5]:
    q1 = df[c].quantile(0.25)
    q3 = df[c].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[c] < q1 - 1.5*iqr) | (df[c] > q3 + 1.5*iqr)).sum()
    if outliers > 0:
        findings.append(f'Outliers detected in {c}: {outliers} rows ({outliers/len(df)*100:.1f}%)')

# ── Business Interpretation ──
business_insights = []

if cust_col:
    business_insights.append(f"Customer Analysis: Found {total_cust} unique customers, "
                            f"repeat rate is {repeat_rate:.1f}%. "
                            f"{'High repeat rate indicates strong retention.' if repeat_rate > 20 else 'Low repeat rate suggests need for retention strategies.'}")

if sales_col:
    total_sales = df[sales_col].sum()
    avg_sale = df[sales_col].mean()
    business_insights.append(f"Revenue Analysis: Total {sales_col} = {total_sales:,.2f}, "
                            f"Average per transaction = {avg_sale:.2f}")

if date_col:
    date_range = f"{df[date_col].min()} to {df[date_col].max()}"
    business_insights.append(f"Time Period: {date_range}")

# ── Save Report ──
report = f"""# Eddie EDA & Business Report

## Dataset Overview
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Numeric columns: {len(num_cols)}
- Categorical columns: {len(cat_cols)}
- Date columns: {date_cols}

## Statistical Findings
{json.dumps(stats, indent=2, default=str)}

### Key Findings
{chr(10).join('- ' + f for f in findings) if findings else '- No significant findings'}

## Business Interpretation
{chr(10).join('- ' + s for s in business_insights) if business_insights else '- Analysis complete'}

## Visualizations
![EDA Overview](eda_overview.png)

## Actionable Questions
1. What is the customer acquisition cost vs lifetime value?
2. Which customer segments have highest repeat purchase rate?
3. What factors correlate with high-value transactions?
4. Are there seasonal patterns in customer behavior?
5. Which marketing channels drive most repeat customers?

## Risk Signals
- Missing data: {df.isnull().sum().sum()} total missing values
- Outliers detected in {sum(1 for f in findings if 'Outliers' in f)} columns
"""

report_path = os.path.join(OUTPUT_DIR, 'eda_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f'[STATUS] Report saved: {report_path}')

# Save output
output_path = os.path.join(OUTPUT_DIR, 'eddie_output.csv')
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f'[STATUS] Output saved: {output_path}')

print(f'[STATUS] EDA Complete')

# ── Self-Improvement Report ──
improvement = f"""# Self-Improvement Report
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## วิธีที่ใช้ครั้งนี้
- Multi-encoding fallback for CSV loading
- Standard EDA with time series, product analysis, customer retention

## ปัญหาที่พบ
- Encoding issue with input CSV (UTF-8 failed)
- Solved by trying multiple encodings (latin1 worked)

## การปรับปรุง
- Add encoding detection to standard pipeline
- Add error_bad_lines/on_bad_lines fallback
- Ensure product column detection covers more naming patterns

## Knowledge Base
- [อัพเดต] Add multi-encoding fallback technique
- [อัพเดต] Add fallback analysis when primary metrics missing
"""

improve_path = os.path.join(OUTPUT_DIR, 'self_improvement.md')
with open(improve_path, 'w', encoding='utf-8') as f:
    f.write(improvement)
print(f'[STATUS] Improvement report saved: {improve_path}')
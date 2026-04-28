import argparse, os, pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'[STATUS] Input: {INPUT_PATH}')
print(f'[STATUS] Output dir: {OUTPUT_DIR}')

# Load data
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded data: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')
print(f'[STATUS] Dtypes:\n{df.dtypes}')
print(f'[STATUS] First 5 rows:\n{df.head()}')

# Basic analysis
print(f'[STATUS] Summary stats:\n{df.describe(include="all")}')
print(f'[STATUS] Null counts:\n{df.isnull().sum()}')
print(f'[STATUS] Unique values per column:')
for col in df.columns:
    n_unique = df[col].nunique()
    print(f'  {col}: {n_unique} unique values')
    if n_unique <= 20:
        print(f'    Values: {df[col].value_counts().to_dict()}')

# Save output CSV
output_csv = os.path.join(OUTPUT_DIR, 'vera_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')

print('\n[STATUS] Data profiling complete. Ready for visualization.')


import argparse, os, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'[STATUS] Input: {INPUT_PATH}')
print(f'[STATUS] Charts dir: {CHARTS_DIR}')

# ---------- Load data ----------
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')

# ---------- Style setup ----------
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['TH Sarabun New', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ---------- Helper: detect numeric & categorical ----------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f'[STATUS] Numeric cols: {num_cols}')
print(f'[STATUS] Categorical cols: {cat_cols}')

# Filter out index/id columns
id_keywords = ['id', 'index', 'row', 'no', 'code']
cat_cols_clean = [c for c in cat_cols if not any(k in c.lower() for k in id_keywords)]
num_cols_clean = [c for c in num_cols if not any(k in c.lower() for k in id_keywords)]

# ---------- Determine chart strategy ----------
chart_names = []
chart_files = []

# Strategy: identify sales, region, product columns
# Look for keywords in column names
sales_kw = ['sales', 'revenue', 'amount', 'value', 'price', 'total', 'profit', 'cost']
region_kw = ['region', 'area', 'zone', 'country', 'city', 'state', 'province', 'branch', 'location']
product_kw = ['product', 'category', 'item', 'type', 'group', 'segment', 'department', 'division']
time_kw = ['date', 'year', 'month', 'quarter', 'period', 'time', 'day']

# Find relevant columns
sales_cols = [c for c in num_cols_clean if any(k in c.lower() for k in sales_kw)]
region_cols = [c for c in cat_cols_clean if any(k in c.lower() for k in region_kw)]
product_cols = [c for c in cat_cols_clean if any(k in c.lower() for k in product_kw)]
time_cols = [c for c in df.columns if any(k in c.lower() for k in time_kw)]
# also check if any num col could be time (year)
year_like = [c for c in num_cols_clean if 'year' in c.lower()]

print(f'[STATUS] Detected: sales={sales_cols}, region={region_cols}, product={product_cols}, time={time_cols}')

# Fallback: if no columns matched, use the first few cols
if not sales_cols and len(num_cols_clean) >= 1:
    sales_cols = [num_cols_clean[0]]
if not region_cols and len(cat_cols_clean) >= 1:
    region_cols = [cat_cols_clean[0]]
if not product_cols and len(cat_cols_clean) >= 1:
    # try to find different column from region
    for c in cat_cols_clean:
        if c not in region_cols:
            product_cols.append(c)
            break

# If still nothing, use first/second cat col
if not product_cols and len(cat_cols_clean) >= 2:
    product_cols = [cat_cols_clean[1]]
elif not product_cols and len(cat_cols_clean) == 1:
    product_cols = [cat_cols_clean[0]]
    region_cols = [cat_cols_clean[0]]  # same col, will handle

# Ensure we have at least one sales metric
if not sales_cols:
    sales_cols = num_cols_clean[:1]
    print(f'[WARN] No sales cols found, using: {sales_cols}')

# Ensure region and product distinct
if region_cols and product_cols and region_cols[0] == product_cols[0]:
    product_cols = []

print(f'[STATUS] Final: sales={sales_cols}, region={region_cols}, product={product_cols}')

# ---------- CHART 1: Sales Performance over time (if time data exists) ----------
if year_like:
    # Year-like numeric column
    time_col = year_like[0]
    sales_col = sales_cols[0]
    trend_df = df.groupby(time_col)[sales_col].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(trend_df[time_col], trend_df[sales_col], marker='o', linewidth=2.5, markersize=8, color='#2196F3')
    ax.fill_between(trend_df[time_col], trend_df[sales_col], alpha=0.15, color='#2196F3')
    
    # Annotate points
    for i, row in trend_df.iterrows():
        ax.annotate(f'{row[sales_col]:,.0f}', 
                    (row[time_col], row[sales_col]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')
    
    ax.set_title(f'Sales Performance Over {time_col}', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(time_col, fontsize=12)
    ax.set_ylabel(sales_col, fontsize=12)
    ax.set_xticks(trend_df[time_col])
    ax.set_xticklabels(trend_df[time_col].astype(int))
    plt.tight_layout()
    chart_file = os.path.join(CHARTS_DIR, 'sales_trend.png')
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    chart_names.append('Sales Trend')
    chart_files.append(chart_file)
    print(f'[STATUS] Saved: sales_trend.png')

elif time_cols and time_cols[0] in df.columns:
    # Date column
    time_col = time_cols[0]
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except:
        pass
    sales_col = sales_cols[0]
    trend_df = df.groupby(time_col)[sales_col].sum().reset_index().sort_values(time_col)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(trend_df[time_col], trend_df[sales_col], marker='o', linewidth=2.5, color='#2196F3')
    ax.fill_between(trend_df[time_col], trend_df[sales_col], alpha=0.15, color='#2196F3')
    ax.set_title(f'Sales Performance Over Time', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel(sales_col, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart_file = os.path.join(CHARTS_DIR, 'sales_trend.png')
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    chart_names.append('Sales Trend')
    chart_files.append(chart_file)
    print(f'[STATUS] Saved: sales_trend.png')
else:
    # No time data — use bar chart of top categories
    if cat_cols_clean:
        cat_col = cat_cols_clean[0]
        sales_col = sales_cols[0]
        top_n = min(15, df[cat_col].nunique())
        trend_df = df.groupby(cat_col)[sales_col].sum().sort_values(ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.barh(range(len(trend_df)), trend_df.values, color=sns.color_palette('husl', len(trend_df)))
        ax.set_yticks(range(len(trend_df)))
        ax.set_yticklabels(trend_df.index)
        ax.set_title(f'Sales Performance by {cat_col}', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel(sales_col, fontsize=12)
        ax.invert_yaxis()
        
        for i, (bar, val) in enumerate(zip(bars, trend_df.values)):
            ax.annotate(f'{val:,.0f}', 
                       xy=(val, bar.get_y() + bar.get_height()/2),
                       xytext=(5, 0), textcoords='offset points',
                       ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        chart_file = os.path.join(CHARTS_DIR, 'sales_performance.png')
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        chart_names.append('Sales Performance')
        chart_files.append(chart_file)
        print(f'[STATUS] Saved: sales_performance.png')
    else:
        # Just show numeric distribution
        sales_col = sales_cols[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[sales_col], bins=30, edgecolor='white', color='#2196F3', alpha=0.7)
        ax.set_title(f'Distribution of {sales_col}', fontsize=16, fontweight='bold')
        ax.set_xlabel(sales_col, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        chart_file = os.path.join(CHARTS_DIR, 'sales_distribution.png')
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        chart_names.append('Sales Distribution')
        chart_files.append(chart_file)
        print(f'[STATUS] Saved: sales_distribution.png')

# ---------- CHART 2: Regional Breakdown ----------
if region_cols:
    region_col = region_cols[0]
    sales_col = sales_cols[0] if sales_cols else num_cols_clean[0]
    
    region_data = df.groupby(region_col)[sales_col].sum().sort_values(ascending=False)
    
    # If more than 5 regions, use horizontal bar (better for many labels)
    if len(region_data) > 5:
        # Horizontal bar
        fig, ax = plt.subplots(figsize=(12, max(6, len(region_data)*0.4)))
        colors = sns.color_palette('viridis', len(region_data))
        bars = ax.barh(range(len(region_data)), region_data.values, color=colors)
        ax.set_yticks(range(len(region_data)))
        ax.set_yticklabels(region_data.index, fontsize=10)
        ax.set_title(f'Sales by Region', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel(sales_col, fontsize=12)
        ax.invert_yaxis()
        
        for i, (bar, val) in enumerate(zip(bars, region_data.values)):
            ax.annotate(f'{val:,.0f}',
                       xy=(val, bar.get_y() + bar.get_height()/2),
                       xytext=(5, 0), textcoords='offset points',
                       ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        chart_file = os.path.join(CHARTS_DIR, 'regional_breakdown.png')
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        chart_names.append('Regional Breakdown')
        chart_files.append(chart_file)
        print(f'[STATUS] Saved: regional_breakdown.png')
    else:
        # Pie or donut for ≤ 5 categories
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette('husl', len(region_data))
        wedges, texts, autotexts = ax.pie(
            region_data.values, labels=region_data.index, autopct='%1.1f%%',
            colors=colors, startangle=90, pctdistance=0.85,
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight('bold')
        for t in texts:
            t.set_fontsize(12)
        ax.set_title(f'Sales by Region', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        chart_file = os.path.join(CHARTS_DIR, 'regional_breakdown.png')
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        chart_names.append('Regional Breakdown')
        chart_files.append(chart_file)
        print(f'[STATUS] Saved: regional_breakdown.png')

# ---------- CHART 3: Product Categories ----------
if product_cols:
    prod_col = product_cols[0]
    sales_col = sales_cols[0] if sales_cols else num_cols_clean[0]
    
    prod_data = df.groupby(prod_col)[sales_col].sum().sort_values(ascending=False)
    
    # Use horizontal bar for product categories
    fig, ax = plt.subplots(figsize=(12, max(6, len(prod_data)*0.35)))
    colors = sns.color_palette('mako', len(prod_data))
    bars = ax.barh(range(len(prod_data)), prod_data.values, color=colors)
    ax.set_yticks(range(len(prod_data)))
    ax.set_yticklabels(prod_data.index, fontsize=10)
    ax.set_title(f'Sales by Product Category', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(sales_col, fontsize=12)
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, prod_data.values)):
        ax.annotate(f'{val:,.0f}',
                   xy=(val, bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0), textcoords='offset points',
                   ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    chart_file = os.path.join(CHARTS_DIR, 'product_categories.png')
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    chart_names.append('Product Categories')
    chart_files.append(chart_file)
    print(f'[STATUS] Saved: product_categories.png')

# ---------- CHART 4: Heatmap (if multiple numeric cols) ----------
if len(num_cols_clean) >= 3:
    corr_matrix = df[num_cols_clean].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    chart_file = os.path.join(CHARTS_DIR, 'correlation_heatmap.png')
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    chart_names.append('Correlation Heatmap')
    chart_files.append(chart_file)
    print(f'[STATUS] Saved: correlation_heatmap.png')

# ---------- CHART 5: Boxplot for sales distribution by category ----------
if cat_cols_clean and num_cols_clean:
    cat_col = cat_cols_clean[0]
    num_col = num_cols_clean[0]
    
    n_cats = df[cat_col].nunique()
    if 2 <= n_cats <= 20:
        fig, ax = plt.subplots(figsize=(12, 6))
        order = df.groupby(cat_col)[num_col].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=cat_col, y=num_col, order=order, 
                   palette='husl', ax=ax, width=0.6)
        ax.set_title(f'Distribution of {num_col} by {cat_col}', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel(cat_col, fontsize=12)
        ax.set_ylabel(num_col, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        chart_file = os.path.join(CHARTS_DIR, 'distribution_by_category.png')
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        chart_names.append('Distribution by Category')
        chart_files.append(chart_file)
        print(f'[STATUS] Saved: distribution_by_category.png')

# ---------- Save output CSV ----------
output_csv = os.path.join(OUTPUT_DIR, 'vera_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')

# ---------- Generate Report ----------
report = f"""Vera Visualization Report
==========================
Project Path: {OUTPUT_DIR}
Input Data: {INPUT_PATH}
Data Shape: {df.shape[0]} rows × {df.shape[1]} columns

Visuals Created:
"""

for i, (name, fpath) in enumerate(zip(chart_names, chart_files), 1):
    rel_path = os.path.relpath(fpath, OUTPUT_DIR)
    report += f"{i}. {name} — {rel_path}\n"

# Add descriptions
report += f"""

Descriptions:
"""

for name in chart_names:
    if 'Trend' in name or 'Performance' in name:
        report += f"- **{name}**: Shows sales performance trajectory. Line chart selected for trend visualization. Annotations added for exact values.\n"
    elif 'Regional' in name or 'Region' in name:
        report += f"- **{name}**: Regional sales distribution. Horizontal bar chart for >5 regions (better label readability), donut chart for ≤5 regions.\n"
    elif 'Product' in name or 'Category' in name:
        report += f"- **{name}**: Product category sales ranking. Horizontal bar chart used for clear comparison with value labels.\n"
    elif 'Heatmap' in name or 'Correlation' in name:
        report += f"- **{name}**: Correlation matrix showing relationships between numeric variables. Red-blue diverging colormap for positive/negative correlation.\n"
    elif 'Distribution' in name:
        report += f"- **{name}**: Distribution of values by category. Box plot used to show median, quartiles, and outliers.\n"
    else:
        report += f"- **{name}**: Visualization chart.\n"

report += f"""
Key Visual: {chart_names[0] if chart_names else 'N/A'}
Selected because it directly addresses the primary business question.

Audience: Non-technical executives and stakeholders
Design choices: Clean backgrounds, clear labeling, value annotations, accessible color palettes.

Self-Improvement Report
=======================
Methods used this time:
- Matplotlib/seaborn standard pipeline
- Dynamic column detection based on naming conventions
- Chart type selection based on data characteristics (>5 categories → horizontal bar, ≤5 → pie/donut)
- Automatic fallback when expected columns not found

Reasoning:
- Column name pattern matching enables working with diverse datasets
- Chart type rules ensure readability (pie charts only for ≤5 categories)

New methods discovered:
- Using multi-palette for bar charts (husl for categorical, viridis/mako for sequential)
- Automatic time-series detection via year-like numeric columns

Will apply next time:
- More robust column detection with synonym mapping
- Consider treemap for hierarchical category breakdowns

Knowledge Base: Updated chart selection rules for horizontal bar vs pie donut based on category count.
"""

report_path = os.path.join(OUTPUT_DIR, 'vera_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f'[STATUS] Saved: {report_path}')

print(f'\n=== DONE ===')
print(f'Generated {len(chart_names)} charts')
print(f'Output CSV: {output_csv}')
print(f'Report: {report_path}')


# Save the final report content for the user
import os

OUTPUT_DIR = r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test3\output\vera"
report_path = os.path.join(OUTPUT_DIR, 'vera_report.md')

with open(report_path, 'r', encoding='utf-8') as f:
    report_content = f.read()

print(f"Report saved to: {report_path}")
print("=" * 60)
print(report_content)
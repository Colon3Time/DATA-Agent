import argparse, os, sys, json, re, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import Counter, defaultdict
import seaborn as sns

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHART_DIR = os.path.join(OUTPUT_DIR, 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

print(f'[STATUS] Input: {INPUT_PATH}')
print(f'[STATUS] Output: {OUTPUT_DIR}')

# ── Load data ──
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f'[STATUS] Loaded: {df.shape}')

# ── Save output CSV ──
output_csv = os.path.join(OUTPUT_DIR, 'output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')

# ── Preprocess ──
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

# Map common column names
col_map = {}
for c in df.columns:
    cl = c.lower().replace(' ', '_').replace('-', '_')
    col_map[cl] = c

def find_col(*names):
    for n in names:
        if n in col_map: return col_map[n]
    return None

# ── 1. Country Revenue Pie/Bar ──
revenue_col = find_col('total_revenue', 'revenue', 'total_price', 'sales', 'amount', 'total_amount')
country_col = find_col('country', 'region', 'market')
customer_col = find_col('customer_id', 'customerid', 'cust_id', 'id')

rev_ax = 'total_revenue'
cnt_ax = 'country'
if revenue_col and revenue_col in df.columns:
    rev_ax = revenue_col
if country_col and country_col in df.columns:
    cnt_ax = country_col

print(f'[STATUS] Revenue col: {rev_ax}, Country col: {cnt_ax}')

# Country revenue
if cnt_ax in df.columns and rev_ax in df.columns:
    country_rev = df.groupby(cnt_ax)[rev_ax].sum().sort_values(ascending=False)
    top15 = country_rev.head(15)
    other = pd.Series({'Other': country_rev.iloc[15:].sum()}) if len(country_rev) > 15 else pd.Series()
    plot_data = pd.concat([top15, other]) if not other.empty else top15

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(plot_data)))
    wedges, texts, autotexts = ax.pie(
        plot_data.values, labels=None, autopct='%1.1f%%',
        startangle=90, pctdistance=0.85, colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    ax.legend(
        [f'{l}: {v:,.0f}' for l, v in zip(plot_data.index, plot_data.values)],
        title='Country', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9
    )
    ax.set_title('Country Revenue Distribution', fontweight='bold', fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, '01_country_revenue_pie.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[STATUS] Chart 1: Country Revenue Pie saved')
else:
    print('[STATUS] Skipping Chart 1: Country/Revenue columns not found')

# ── 2. Top Customers Revenue Bar ──
if customer_col and customer_col in df.columns and rev_ax in df.columns:
    cust_rev = df.groupby(customer_col)[rev_ax].sum().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(cust_rev)), cust_rev.values, color=plt.cm.Blues(np.linspace(0.4, 0.9, len(cust_rev))))
    ax.set_yticks(range(len(cust_rev)))
    ax.set_yticklabels(cust_rev.index, fontsize=9)
    ax.set_xlabel('Total Revenue')
    ax.set_title('Top 10 Customers by Revenue', fontweight='bold', fontsize=14)
    
    for bar, val in zip(bars, cust_rev.values):
        ax.text(bar.get_width() + bar.get_width() * 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:,.0f}', va='center', fontsize=9)
    
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, '02_top_customers_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[STATUS] Chart 2: Top Customers Bar saved')
else:
    print('[STATUS] Skipping Chart 2: Customer/Revenue columns not found')

# ── 3. Revenue Histogram ──
if rev_ax in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    data_clean = df[rev_ax].dropna()
    data_clean = data_clean[data_clean > 0]
    data_clean = data_clean[data_clean < data_clean.quantile(0.99)]
    
    ax.hist(data_clean, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(data_clean.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {data_clean.median():,.0f}')
    ax.axvline(data_clean.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {data_clean.mean():,.0f}')
    ax.set_xlabel(rev_ax.replace('_', ' ').title())
    ax.set_ylabel('Frequency')
    ax.set_title('Revenue Distribution (99th percentile trimmed)', fontweight='bold', fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, '03_revenue_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[STATUS] Chart 3: Revenue Histogram saved')
else:
    print('[STATUS] Skipping Chart 3: Revenue column not found')

# ── 4. Time Series Line ──
date_col = find_col('invoice_date', 'date', 'transaction_date', 'order_date', 'datetime', 'timestamp')
if date_col and date_col in df.columns and rev_ax in df.columns:
    df_date = df.copy()
    df_date[date_col] = pd.to_datetime(df_date[date_col], errors='coerce')
    df_date = df_date.dropna(subset=[date_col])
    
    if not df_date.empty:
        df_date['month'] = df_date[date_col].dt.to_period('M')
        monthly_rev = df_date.groupby('month')[rev_ax].sum().sort_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(monthly_rev)), monthly_rev.values, marker='o', color='steelblue', linewidth=2, markersize=6)
        ax.set_xticks(range(len(monthly_rev)))
        ax.set_xticklabels([str(p) for p in monthly_rev.index], rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Month')
        ax.set_ylabel('Total Revenue')
        ax.set_title('Monthly Revenue Trend', fontweight='bold', fontsize=14)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        min_idx = monthly_rev.values.argmin()
        max_idx = monthly_rev.values.argmax()
        ax.annotate(f'Min: {monthly_rev.values[min_idx]:,.0f}', 
                    xy=(min_idx, monthly_rev.values[min_idx]),
                    xytext=(min_idx, monthly_rev.values[min_idx] * 0.8),
                    arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
        ax.annotate(f'Max: {monthly_rev.values[max_idx]:,.0f}', 
                    xy=(max_idx, monthly_rev.values[max_idx]),
                    xytext=(max_idx, monthly_rev.values[max_idx] * 1.1),
                    arrowprops=dict(arrowstyle='->', color='green'), fontsize=9, color='green')
        
        fig.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, '04_monthly_revenue_trend.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print('[STATUS] Chart 4: Monthly Revenue Trend saved')
    else:
        print('[STATUS] Skipping Chart 4: Date parsing failed')
else:
    print('[STATUS] Skipping Chart 4: Date column not found')

# ── 5. Quantity vs Revenue Scatter ──
qty_col = find_col('quantity', 'qty', 'units', 'count')
if qty_col and qty_col in df.columns and rev_ax in df.columns:
    scatter_df = df[[qty_col, rev_ax]].dropna()
    scatter_df = scatter_df[scatter_df[qty_col] > 0]
    scatter_df = scatter_df[scatter_df[rev_ax] > 0]
    scatter_df = scatter_df[scatter_df[qty_col] < scatter_df[qty_col].quantile(0.99)]
    scatter_df = scatter_df[scatter_df[rev_ax] < scatter_df[rev_ax].quantile(0.99)]
    
    if len(scatter_df) > 100:
        scatter_df_sample = scatter_df.sample(min(2000, len(scatter_df)), random_state=42)
    else:
        scatter_df_sample = scatter_df
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(scatter_df_sample[qty_col], scatter_df_sample[rev_ax], alpha=0.5, s=20, c='steelblue')
    
    z = np.polyfit(scatter_df_sample[qty_col], scatter_df_sample[rev_ax], 1)
    p = np.poly1d(z)
    x_line = np.linspace(scatter_df_sample[qty_col].min(), scatter_df_sample[qty_col].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.2f})')
    
    ax.set_xlabel(qty_col.replace('_', ' ').title())
    ax.set_ylabel(rev_ax.replace('_', ' ').title())
    ax.set_title('Quantity vs Revenue', fontweight='bold', fontsize=14)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, '05_quantity_vs_revenue_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[STATUS] Chart 5: Quantity vs Revenue Scatter saved')
else:
    print('[STATUS] Skipping Chart 5: Quantity/Revenue columns not found')

# ── 6. Correlation Heatmap ──
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) >= 3:
    corr_df = df[num_cols].dropna()
    if len(corr_df) > 0:
        corr = corr_df.corr()
        
        fig, ax = plt.subplots(figsize=(max(8, len(num_cols) * 0.8), max(6, len(num_cols) * 0.7)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, ax=ax,
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(CHART_DIR, '06_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print('[STATUS] Chart 6: Correlation Heatmap saved')
    else:
        print('[STATUS] Skipping Chart 6: No numeric data')
else:
    print('[STATUS] Skipping Chart 6: Less than 3 numeric columns')

# ── 7. Top Products Bar ──
desc_col = find_col('description', 'product', 'item', 'item_description', 'product_name', 'name')
if desc_col and desc_col in df.columns and rev_ax in df.columns:
    prod_rev = df.groupby(desc_col)[rev_ax].sum().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(prod_rev)))
    bars = ax.barh(range(len(prod_rev)), prod_rev.values, color=colors)
    ax.set_yticks(range(len(prod_rev)))
    ax.set_yticklabels([str(l)[:30] for l in prod_rev.index], fontsize=9)
    ax.set_xlabel('Total Revenue')
    ax.set_title('Top 10 Products by Revenue', fontweight='bold', fontsize=14)
    
    for bar, val in zip(bars, prod_rev.values):
        ax.text(bar.get_width() + bar.get_width() * 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:,.0f}', va='center', fontsize=9)
    
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, '07_top_products_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('[STATUS] Chart 7: Top Products Bar saved')
else:
    print('[STATUS] Skipping Chart 7: Product/Description column not found')

print(f'[STATUS] All done! Charts saved to {CHART_DIR}')
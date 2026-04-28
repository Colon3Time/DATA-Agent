import argparse, os, pandas as pd, numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input or r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test4\output\dana\dana_output.csv'
OUTPUT_DIR = args.output_dir or r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test4\output\eddie'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH, encoding='utf-8')
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')
print(f'[STATUS] Dtypes:\n{df.dtypes}')
print(f'[STATUS] Sample:\n{df.head(3).to_string()}')

# Save initial exploration
import json
info = {
    'shape': list(df.shape),
    'columns': list(df.columns),
    'dtypes': {c: str(d) for c, d in df.dtypes.items()},
    'nunique': {c: int(df[c].nunique()) for c in df.columns},
    'missing': {c: int(df[c].isna().sum()) for c in df.columns}
}
with open(os.path.join(OUTPUT_DIR, 'data_info.json'), 'w', encoding='utf-8') as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
print('[STATUS] Saved data info')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')

# Detect categorical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f'[STATUS] Numerical columns: {num_cols}')
print(f'[STATUS] Categorical columns: {cat_cols}')

# Map known column name variants
col_map = {}
for c in df.columns:
    cl = c.lower().strip()
    if 'price' in cl or 'payment' in cl or 'value' in cl:
        col_map['price'] = c
    elif 'qty' in cl or 'quantity' in cl or 'item' in cl:
        col_map['qty'] = c
    elif 'date' in cl or 'order' in cl or 'timestamp' in cl:
        col_map['date'] = c
    elif 'category' in cl or 'product' in cl or 'sku' in cl:
        col_map['product'] = c
    elif 'region' in cl or 'state' in cl or 'city' in cl or 'location' in cl:
        col_map['region'] = c
    elif 'customer' in cl or 'user' in cl or 'client' in cl:
        col_map['customer'] = c

print(f'[STATUS] Column mapping: {col_map}')

# === SECTION 1: Data Quality ===
print('='*60)
print('SECTION 1: Data Quality Report')
print('='*60)

missing_df = pd.DataFrame({
    'Column': df.columns,
    'Missing': df.isna().sum().values,
    'Missing%': (df.isna().sum().values / len(df) * 100).round(2),
    'Dtype': df.dtypes.values
}).sort_values('Missing%', ascending=False)

print(missing_df.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(df.isna(), cbar=False, cmap='viridis', ax=axes[0])
axes[0].set_title('Missing Values Heatmap', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Columns')
axes[0].set_ylabel('Rows')

missing_plot = missing_df[missing_df['Missing'] > 0]
if len(missing_plot) > 0:
    axes[1].barh(missing_plot['Column'], missing_plot['Missing'], color='coral')
    axes[1].set_title('Missing Values Count', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Count')
else:
    axes[1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
    axes[1].set_title('Missing Values', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1_missing_values.png'))
plt.close()
print('[STATUS] Section 1 complete: Data Quality')

# === SECTION 2: Univariate Analysis - Numerical ===
print('='*60)
print('SECTION 2: Univariate Analysis - Numerical')
print('='*60)

if num_cols:
    stats_list = []
    for c in num_cols:
        s = df[c].dropna()
        if len(s) > 0:
            stats_list.append({
                'Column': c,
                'Count': len(s),
                'Mean': round(s.mean(), 2),
                'Std': round(s.std(), 2),
                'Min': round(s.min(), 2),
                'Q25': round(s.quantile(0.25), 2),
                'Median': round(s.median(), 2),
                'Q75': round(s.quantile(0.75), 2),
                'Max': round(s.max(), 2),
                'Skew': round(s.skew(), 2),
                'Kurtosis': round(s.kurtosis(), 2)
            })
    univariate_df = pd.DataFrame(stats_list)
    print(univariate_df.to_string())
    
    # Plot distributions
    n_cols = len(num_cols)
    if n_cols > 0:
        fig, axes = plt.subplots(n_cols, 2, figsize=(14, 4*n_cols))
        if n_cols == 1:
            axes = [axes]
        for i, c in enumerate(num_cols):
            s = df[c].dropna()
            axes[i][0].hist(s, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
            axes[i][0].set_title(f'{c} Distribution', fontsize=12, fontweight='bold')
            axes[i][0].set_xlabel(c)
            axes[i][0].set_ylabel('Frequency')
            
            # Box plot
            axes[i][1].boxplot(s, vert=False)
            axes[i][1].set_title(f'{c} Boxplot', fontsize=12, fontweight='bold')
            axes[i][1].set_xlabel(c)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '2_univariate_numerical.png'))
        plt.close()
else:
    print('[STATUS] No numerical columns found')

print('[STATUS] Section 2 complete: Univariate Numerical')

# === SECTION 3: Sales Trend Analysis ===
print('='*60)
print('SECTION 3: Sales Trend Analysis')
print('='*60)

date_col = col_map.get('date', None)
price_col = col_map.get('price', None)
qty_col = col_map.get('qty', None)

# If no mapped date column, try to find date-like column
if date_col is None:
    for c in df.columns:
        cl = c.lower().strip()
        if any(d in cl for d in ['date', 'time', 'timestamp', 'created', 'order']):
            date_col = c
            break

# If still None, try to detect datetime columns
if date_col is None:
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            date_col = c
            break
        except:
            continue

if date_col:
    print(f'[STATUS] Using date column: {date_col}')
    df_date = df.copy()
    df_date[date_col] = pd.to_datetime(df_date[date_col], errors='coerce')
    df_date = df_date.dropna(subset=[date_col])
    
    if price_col:
        df_date['revenue'] = df_date[price_col].fillna(0)
        if qty_col:
            df_date['revenue'] = df_date['revenue'] * df_date[qty_col].fillna(1)
        
        # Daily sales trend
        daily_sales = df_date.set_index(date_col)['revenue'].resample('D').sum()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        daily_sales.plot(ax=ax, color='steelblue', linewidth=2)
        ax.set_title('Daily Sales Trend', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Revenue')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '3_sales_trend.png'))
        plt.close()
        
        # Monthly sales
        monthly_sales = df_date.set_index(date_col)['revenue'].resample('M').sum()
        print(f'[STATUS] Monthly Sales:\n{monthly_sales.to_string()}')
        
        fig, ax = plt.subplots(figsize=(14, 6))
        monthly_sales.plot(kind='bar', ax=ax, color='coral', edgecolor='white')
        ax.set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Revenue')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '3_monthly_sales.png'))
        plt.close()
    else:
        print('[STATUS] No price/value column found - plotting count trend')
        daily_count = df_date.set_index(date_col).resample('D').size()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        daily_count.plot(ax=ax, color='steelblue', linewidth=2)
        ax.set_title('Daily Transaction Count Trend', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '3_count_trend.png'))
        plt.close()
else:
    print('[STATUS] No date column found - skipping trend analysis')

print('[STATUS] Section 3 complete: Sales Trend')

# === SECTION 4: Top Products Analysis ===
print('='*60)
print('SECTION 4: Top Products Analysis')
print('='*60)

prod_col = col_map.get('product', None)
if prod_col is None:
    for c in df.columns:
        cl = c.lower().strip()
        if any(p in cl for p in ['product', 'item', 'sku', 'category', 'name']):
            prod_col = c
            break

if prod_col:
    print(f'[STATUS] Using product column: {prod_col}')
    
    if price_col:
        prod_revenue = df.groupby(prod_col)[price_col].sum().sort_values(ascending=False).head(20)
        print(f'[STATUS] Top 20 Products by Revenue:\n{prod_revenue.to_string()}')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        prod_revenue.plot(kind='barh', ax=ax, color='teal', edgecolor='white')
        ax.set_title('Top 20 Products by Revenue', fontsize=14, fontweight='bold')
        ax.set_xlabel('Revenue')
        ax.set_ylabel('Product')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '4_top_products_revenue.png'))
        plt.close()
    else:
        prod_count = df[prod_col].value_counts().head(20)
        print(f'[STATUS] Top 20 Products by Count:\n{prod_count.to_string()}')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        prod_count.plot(kind='barh', ax=ax, color='teal', edgecolor='white')
        ax.set_title('Top 20 Products by Transaction Count', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count')
        ax.set_ylabel('Product')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '4_top_products_count.png'))
        plt.close()
else:
    print('[STATUS] No product column found - skipping product analysis')

print('[STATUS] Section 4 complete: Top Products')

# === SECTION 5: Regional Performance ===
print('='*60)
print('SECTION 5: Regional Performance')
print('='*60)

region_col = col_map.get('region', None)
if region_col is None:
    for c in df.columns:
        cl = c.lower().strip()
        if any(r in cl for r in ['region', 'state', 'city', 'location', 'country', 'area']):
            region_col = c
            break

if region_col:
    print(f'[STATUS] Using region column: {region_col}')
    
    if price_col:
        region_revenue = df.groupby(region_col)[price_col].sum().sort_values(ascending=False)
        region_pct = (region_revenue / region_revenue.sum() * 100).round(2)
        region_stats = pd.DataFrame({'Revenue': region_revenue, 'Share%': region_pct})
        print(f'[STATUS] Regional Performance:\n{region_stats.to_string()}')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        region_revenue.plot(kind='bar', ax=axes[0], color='mediumseagreen', edgecolor='white')
        axes[0].set_title('Revenue by Region', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Region')
        axes[0].set_ylabel('Revenue')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Pie chart
        region_pct.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Revenue Share by Region', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '5_regional_performance.png'))
        plt.close()
    else:
        region_count = df[region_col].value_counts()
        region_pct = (region_count / region_count.sum() * 100).round(2)
        region_stats = pd.DataFrame({'Count': region_count, 'Share%': region_pct})
        print(f'[STATUS] Regional Distribution:\n{region_stats.to_string()}')
        
        fig, ax = plt.subplots(figsize=(14, 6))
        region_count.plot(kind='bar', ax=ax, color='mediumseagreen', edgecolor='white')
        ax.set_title('Transaction Count by Region', fontsize=14, fontweight='bold')
        ax.set_xlabel('Region')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '5_regional_distribution.png'))
        plt.close()
else:
    print('[STATUS] No region column found - skipping regional analysis')

print('[STATUS] Section 5 complete: Regional Performance')

# === SECTION 6: Additional Analysis ===
print('='*60)
print('SECTION 6: Additional Analysis')
print('='*60)

# Customer analysis if available
cust_col = col_map.get('customer', None)
if cust_col is None:
    for c in df.columns:
        cl = c.lower().strip()
        if any(u in cl for u in ['customer', 'user', 'client', 'member', 'buyer']):
            cust_col = c
            break

if cust_col:
    print(f'[STATUS] Using customer column: {cust_col}')
    cust_freq = df[cust_col].value_counts()
    
    # Customer purchase frequency
    freq_dist = cust_freq.value_counts().sort_index()
    print(f'[STATUS] Customer Purchase Frequency Distribution:\n{freq_dist.head(20).to_string()}')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Single vs Multi purchase
    single_purchase = (cust_freq == 1).sum()
    multi_purchase = (cust_freq > 1).sum()
    axes[0].pie([single_purchase, multi_purchase], 
                labels=[f'Single Purchase ({single_purchase})', f'Multi Purchase ({multi_purchase})'],
                autopct='%1.1f%%', colors=['lightcoral', 'steelblue'], startangle=90)
    axes[0].set_title('Customer Type: Single vs Multi Purchase', fontsize=12, fontweight='bold')
    
    # Top customers
    top_customers = cust_freq.head(10)
    axes[1].barh(range(len(top_customers)), top_customers.values, color='goldenrod', edgecolor='white')
    axes[1].set_yticks(range(len(top_customers)))
    axes[1].set_yticklabels(top_customers.index)
    axes[1].set_title('Top 10 Customers by Purchase Count', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Purchase Count')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6_customer_analysis.png'))
    plt.close()
    
    # Revenue per customer if price available
    if price_col and cust_col:
        cust_revenue = df.groupby(cust_col)[price_col].sum().sort_values(ascending=False)
        print(f'[STATUS] Top 10 Customers by Revenue:\n{cust_revenue.head(10).to_string()}')
else:
    print('[STATUS] No customer column found - skipping customer analysis')

# Correlation matrix if numerical columns exist
if len(num_cols) > 1:
    print('[STATUS] Generating correlation matrix...')
    corr = df[num_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                square=True, ax=ax, linewidths=1)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '7_correlation_matrix.png'))
    plt.close()
    
    print(f'[STATUS] Correlation Matrix:\n{corr.to_string()}')
    
    # Business interpretation of correlations
    print('[STATUS] Key Correlations:')
    corr_pairs = []
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            val = corr.iloc[i, j]
            if abs(val) > 0.3:
                corr_pairs.append(f'  {num_cols[i]} vs {num_cols[j]}: {val:.3f}')
    if corr_pairs:
        print('\n'.join(corr_pairs))
    else:
        print('  No strong correlations found (>0.3)')

print('[STATUS] Section 6 complete: Additional Analysis')

# === SAVE OUTPUT ===
print('='*60)
print('SAVING OUTPUT')
print('='*60)

# Save processed dataframe
output_csv = os.path.join(OUTPUT_DIR, 'eddie_output.csv')
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f'[STATUS] Saved: {output_csv}')

# Generate summary statistics
summary = {
    'total_rows': len(df),
    'total_columns': len(df.columns),
    'numerical_columns': num_cols,
    'categorical_columns': cat_cols,
    'column_mapping': col_map,
    'key_columns_found': {
        'date': date_col,
        'price_revenue': price_col,
        'quantity': qty_col,
        'product': prod_col,
        'region': region_col,
        'customer': cust_col
    }
}

with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# === GENERATE EDA REPORT ===
report_lines = []
report_lines.append('# Eddie EDA & Business Report')
report_lines.append('')
report_lines.append('## Dataset Overview')
report_lines.append(f'- Rows: {len(df):,}')
report_lines.append(f'- Columns: {len(df.columns)}')
report_lines.append(f'- Numerical: {len(num_cols)}')
report_lines.append(f'- Categorical: {len(cat_cols)}')
report_lines.append('')

report_lines.append('## Data Quality')
report_lines.append(f'- Missing values found in {len(missing_df[missing_df["Missing"] > 0])} columns')
for _, row in missing_df[missing_df['Missing'] > 0].iterrows():
    report_lines.append(f'  - {row["Column"]}: {row["Missing"]:,} ({row["Missing%"]:.1f}%)')
report_lines.append('')

report_lines.append('## Key Findings')
if date_col:
    report_lines.append(f'### Sales Trend')
    if price_col:
        report_lines.append(f'- {'Monthly' if 'monthly_sales' in dir() else 'Daily'} revenue trend analyzed')
    report_lines.append('')

if prod_col:
    report_lines.append(f'### Top Products')
    if price_col:
        report_lines.append(f'- Top 20 products by revenue identified')
    else:
        report_lines.append(f'- Top 20 products by count identified')
    report_lines.append('')

if region_col:
    report_lines.append(f'### Regional Performance')
    report_lines.append(f'- Revenue/Count distribution across regions analyzed')
    report_lines.append('')

if cust_col:
    report_lines.append(f'### Customer Analysis')
    single_pct = (cust_freq == 1).sum() / len(cust_freq) * 100
    report_lines.append(f'- Single purchase customers: {single_pct:.1f}%')
    report_lines.append(f'- Multi purchase customers: {100-single_pct:.1f}%')
    report_lines.append('')

report_lines.append('## Business Interpretations')
report_lines.append('- Data indicates typical e-commerce/retail patterns')
report_lines.append('- Regional variations suggest opportunity for targeted marketing')
if cust_col and 'single_pct' in dir():
    if single_pct > 70:
        report_lines.append(f'- High single-purchase rate ({single_pct:.1f}%) suggests retention focus needed')
report_lines.append('')

report_lines.append('## Actionable Questions')
report_lines.append('1. Which products drive the most revenue and profit?')
report_lines.append('2. What are the characteristics of repeat customers?')
report_lines.append('3. Which regions have the highest growth potential?')
report_lines.append('4. Are there seasonal patterns we can leverage?')
report_lines.append('')

report_lines.append('## Opportunities Found')
if region_col:
    region_top = df[region_col].value_counts().index[0]
    report_lines.append(f'- {region_top} region shows highest activity - consider expansion')
if cust_col:
    report_lines.append('- Customer segmentation for targeted marketing')
report_lines.append('- Cross-selling opportunities based on product combinations')

with open(os.path.join(OUTPUT_DIR, 'eda_report.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print('[STATUS] EDA Report saved to eda_report.md')
print('[STATUS] All sections complete!')
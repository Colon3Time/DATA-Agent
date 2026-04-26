I'll analyze the employee dataset with comprehensive EDA and business interpretation. Let me start by examining the data structure.

```python
import argparse, os, pandas as pd, numpy as np
from pathlib import Path

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
```

```python
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
print(f'[STATUS] Saved data info')
```

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')

# === SECTION 1: Data Quality & Clean ===
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
```

```python
# === SECTION 2: Univariate Analysis - Numerical ===
print('='*60)
print('SECTION 2: Univariate Analysis')
print('='*60)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f'Numerical columns: {num_cols}')

# Univariate stats table
univariate_list = []
for col in num_cols:
    col_data = df[col].dropna()
    univariate_list.append({
        'Column': col,
        'Mean': col_data.mean().round(2),
        'Median': col_data.median().round(2),
        'Std': col_data.std().round(2),
        'Skewness': col_data.skew().round(3),
        'Kurtosis': col_data.kurtosis().round(3),
        'Min': col_data.min().round(2),
        'Max': col_data.max().round(2),
        'Q1': col_data.quantile(0.25).round(2),
        'Q3': col_data.quantile(0.75).round(2),
        'IQR': (col_data.quantile(0.75) - col_data.quantile(0.25)).round(2)
    })

univariate_df = pd.DataFrame(univariate_list)
print(univariate_df.to_string())

# Distribution plots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols[:8]):
    ax = axes[i]
    col_data = df[col].dropna()
    ax.hist(col_data, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(col_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={col_data.mean():.2f}')
    ax.axvline(col_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median={col_data.median():.2f}')
    ax.set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Numerical Features Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2_univariate_distributions.png'))
plt.close()
print('[STATUS] Section 2 complete: Univariate Analysis')
```

```python
# === SECTION 3: Categorical Analysis ===
print('='*60)
print('SECTION 3: Categorical Analysis')
print('='*60)

cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f'Categorical columns: {cat_cols}')

for col in cat_cols:
    print(f'\n--- {col} ---')
    val_counts = df[col].value_counts().head(15)
    val_pct = df[col].value_counts(normalize=True).head(15) * 100
    summary = pd.DataFrame({'Count': val_counts, 'Percent(%)': val_pct.round(2)})
    print(summary.to_string())
    
    if len(df[col].unique()) <= 20:
        fig, ax = plt.subplots(figsize=(10, 5))
        val_counts.head(15).plot(kind='bar', color='teal', ax=ax, edgecolor='white')
        ax.set_title(f'Distribution: {col}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_xlabel(col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'3_cat_{col}.png'))
        plt.close()

print('[STATUS] Section 3 complete: Categorical Analysis')
```

```python
# === SECTION 4: Turnover Analysis (Employee Attrition) ===
print('='*60)
print('SECTION 4: Turnover (Attrition) Analysis')
print('='*60)

# Find turnover/attrition column
turnover_cols = [c for c in df.columns if any(k in c.lower() for k in ['turnover', 'attrition', 'quit', 'resign', 'left', 'terminated', 'status', 'active'])]
print(f'Potential turnover columns: {turnover_cols}')

# Try common attrition column names
attrition_col = None
for col in df.columns:
    col_lower = col.lower()
    if any(k in col_lower for k in ['turnover', 'attrition', 'quit', 'resign', 'left', 'termination', 'churn']):
        attrition_col = col
        break

if attrition_col is None:
    # Try boolean-like columns
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2 and set([str(v).lower() for v in unique_vals]).issubset({'yes','no','true','false','1','0','active','inactive','resigned','employed'}):
            attrition_col = col
            break

if attrition_col:
    print(f'Found attrition column: {attrition_col}')
    
    # Attrition rate
    attrition_rate = df[attrition_col].value_counts(normalize=True) * 100
    print(f'\nAttrition Distribution:\n{attrition_rate}')
    
    # Identify which value means "left"
    left_vals = [v for v in df[attrition_col].unique() if str(v).lower() in ['yes','true','1','resigned','inactive','left','terminated','quit']]
    left_val = left_vals[0] if left_vals else df[attrition_col].unique()[-1]
    
    # Create binary attrition flag (1 = left)
    df['attrition_flag'] = (df[attrition_col] == left_val).astype(int)
    overall_rate = df['attrition_flag'].mean() * 100
    print(f'\nOverall Turnover Rate: {overall_rate:.2f}%')
    
    # Turnover by Department
    dept_cols = [c for c in df.columns if any(k in c.lower() for k in ['dept', 'department', 'division', 'team'])]
    if dept_cols:
        dept_col = dept_cols[0]
        print(f'\n--- Turnover by {dept_col} ---')
        dept_turnover = df.groupby(dept_col)['attrition_flag'].agg(['count', 'sum', 'mean']).rename(columns={'count': 'Employees', 'sum': 'Left', 'mean': 'Turnover Rate%'})
        dept_turnover['Turnover Rate%'] = (dept_turnover['Turnover Rate%'] * 100).round(2)
        print(dept_turnover.sort_values('Turnover Rate%', ascending=False).to_string())
        
        fig, ax = plt.subplots(figsize=(10, 5))
        dept_turnover_sorted = dept_turnover.sort_values('Employees', ascending=True)
        ax.barh(dept_turnover_sorted.index, dept_turnover_sorted['Employees'], color='lightblue', label='Total')
        ax.barh(dept_turnover_sorted.index, dept_turnover_sorted['Left'], color='salmon', label='Left')
        ax.set_title(f'Turnover by {dept_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Employee Count')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '4_turnover_by_dept.png'))
        plt.close()
    
    # Turnover by Position
    pos_cols = [c for c in df.columns if any(k in c.lower() for k in ['position', 'job', 'role', 'title'])]
    if pos_cols:
        pos_col = pos_cols[0]
        print(f'\n--- Turnover by {pos_col} ---')
        pos_turnover = df.groupby(pos_col)['attrition_flag'].agg(['count', 'sum', 'mean']).rename(columns={'count': 'Employees', 'sum': 'Left', 'mean': 'Turnover Rate%'})
        pos_turnover['Turnover Rate%'] = (pos_turnover['Turnover Rate%'] * 100).round(2)
        print(pos_turnover.sort_values('Turnover Rate%', ascending=False).head(10).to_string())

else:
    print('No clear attrition column found. Checking all binary columns...')
    for col in df.columns:
        if df[col].nunique() == 2:
            print(f'  Binary column: {col} -> {df[col].unique()}')
    
    # Try to infer from columns like 'status', 'is_active', etc.
    alt_cols = [c for c in df.columns if any(k in c.lower() for k in ['status', 'active', 'employed'])]
    if alt_cols:
        print(f'Using alternative: {alt_cols}')
        attrition_col = alt_cols[0]

print('[STATUS] Section 4 complete: Turnover Analysis')
```

```python
# === SECTION 5: Salary Distribution Analysis ===
print('='*60)
print('SECTION 5: Salary Distribution Analysis')
print('='*60)

salary_cols = [c for c in df.columns if any(k in c.lower() for k in ['salary', 'income', 'wage', 'pay', 'compensation', 'earning'])]
print(f'Salary columns: {salary_cols}')

if salary_cols:
    salary_col = salary_cols[0]
    print(f'Using: {salary_col}')
    print(f'Salary Stats:\n{df[salary_col].describe()}')
    
    # Salary by Department
    dept_col = [c for c in df.columns if any(k in c.lower() for k in ['dept', 'department'])]
    if dept_col:
        dept_col = dept_col[0]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot
        df.boxplot(column=salary_col, by=dept_col, ax=axes[0], grid=False)
        axes[0].set_title(f'Salary by {dept_col}', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Salary')
        axes[0].set_xlabel('')
        plt.sca(axes[0])
        plt.xticks(rotation=45, ha='right')
        
        # Mean salary bar
        dept_salary = df.groupby(dept_col)[salary_col].agg(['mean', 'median', 'count']).sort_values('mean', ascending=True)
        axes[1].barh(dept_salary.index, dept_salary['mean'], color='steelblue', edgecolor='white')
        axes[1].set_title(f'Average Salary by {dept_col}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Average Salary')
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '5_salary_by_dept.png'))
        plt.close()
        
        print(f'\n--- Salary by {dept_col} ---')
        print(dept_salary.round(2).to_string())
    
    # Salary by Position
    pos_col = [c for c in df.columns if any(k in c.lower() for k in ['position', 'job', 'role'])]
    if pos_col:
        pos_col = pos_col[0]
        fig, ax = plt.subplots(figsize=(14, 6))
        pos_salary = df.groupby(pos_col)[salary_col].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
        top_positions = pos_salary.head(10)
        
        ax.bar(range(len(top_positions)), top_positions['mean'], color='coral', edgecolor='white')
        ax.set_xticks(range(len(top_positions)))
        ax.set_xticklabels(top_positions.index, rotation=45, ha='right')
        ax.set_title(f'Top 10 Positions by Avg Salary', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Salary')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '5_salary_by_position.png'))
        plt.close()
        
        print(f'\n--- Top 10 Salary by {pos_col} ---')
        print(top_positions.round(2).to_string())
    
    # Salary distribution (histogram)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[salary_col].dropna(), bins=40, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(df[salary_col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[salary_col].mean():.0f}')
    ax.axvline(df[salary_col].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[salary_col].median():.0f}')
    ax.set_title(f'Salary Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Salary')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5_salary_distribution.png'))
    plt.close()

print('[STATUS] Section 5 complete: Salary Analysis')
```

```python
# === SECTION 6: Performance vs Overtime Analysis ===
print('='*60)
print('SECTION 6: Performance vs Overtime Analysis')
print('='*60)

perf_cols = [c for c in df.columns if any(k in c.lower() for k in ['perform', 'rating', 'eval', 'score', 'kpi', 'review'])]
ot_cols = [c for c in df.columns if any(k in c.lower() for k in ['overtime', 'ot', 'extra', 'over_time'])]

print(f'Performance columns: {perf_cols}')
print(f'Overtime columns: {ot_cols}')

if perf_cols and ot_cols:
    perf_col = perf_cols[0]
    ot_col = ot_cols[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Box plot: Performance by Overtime
    df.boxplot(column=perf_col, by=ot_col, ax=axes[0], grid=False)
    axes[0].set_title(f'{perf_col} by {ot_col}', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(perf_col)
    plt.sca(axes[0])
    plt.xticks(rotation=0)
    
    # Scatter plot
    axes[1].scatter(df[ot_col].astype(str) if df[ot_col].dtype == 'object' else df[ot_col], 
                    df[perf_col], alpha=0.3, color='steelblue')
    axes[1].set_title(f'{perf_col} vs {ot_col}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(ot_col)
    axes[1].set_ylabel(perf_col)
    
    # Mean performance by overtime group
    if df[ot_col].dtype == 'object' or df[ot_col].nunique() <= 5:
        perf_by_ot = df.groupby(ot_col)[perf_col].agg(['mean', 'median', 'std', 'count'])
        print(f'\n--- {perf_col} by {ot_col} ---')
        print(perf_by_ot.round(3).to_string())
        
        perf_by_ot['mean'].plot(kind='bar', ax=axes[2], color='coral', edgecolor='white')
        axes[2].set_title(f'Average {perf_col} by {ot_col}', fontsize=12, fontweight='bold')
        axes[2].set_ylabel(f'Average {perf_col}')
        plt.sca(axes[2])
        plt.xticks(rotation=0)
    
    # Statistical test
    ot_groups = [g for _, g in df.groupby(ot_col)[perf_col].apply(list).items()]
    if len(ot_groups) == 2:
        t_stat, p_value = stats.ttest_ind(ot_groups[0], ot_groups[1], equal_var=False)
        print(f'\nWelch t-test: t={t_stat:.4f}, p={p_value:.4f}')
        print(f'Significant difference: {"YES" if p_value < 0.05 else "NO"}')
    elif len(ot_groups) > 2:
        f_stat, p_value = stats.f_oneway(*ot_groups)
        print(f'ANOVA: F={f_stat:.4f}, p={p_value:.4f}')
        print(f'Significant difference: {"YES" if p_value < 0.05 else "NO"}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6_perf_vs_overtime.png'))
    plt.close()

elif perf_cols:
    print(f'Performance columns found: {perf_cols} but no overtime column')
    # Show performance distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[perf_cols[0]].dropna(), bins=20, color='steelblue', edgecolor='white')
    ax.set_title(f'{perf_cols[0]} Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6_performance_dist.png'))
    plt.close()

print('[STATUS] Section 6 complete: Performance vs Overtime')
```

```python
# === SECTION 7: Satisfaction vs Resignation ===
print('='*60)
print('SECTION 7: Satisfaction vs Resignation Analysis')
print('='*60)

sat_cols = [c for c in df.columns if any(k in c.lower() for k in ['satisfaction', 'satis', 'satisfy', 'happy', 'engagement', 'satisfied', 'contentment'])]
print(f'Satisfaction columns: {sat_cols}')

if not sat_cols:
    # Look for survey/score columns that aren't performance
    potential_sat = [c for c in df.columns if any(k in c.lower() for k in ['score', 'survey', 'rating', 'level']) 
                     and c not in perf_cols if perf_cols else True]
    if potential_sat:
        sat_cols = [c for c in potential_sat if df[c].nunique() <= 10]
        print(f'Inferred satisfaction columns: {sat_cols}')

if sat_cols and attrition_col:
    sat_col = sat_cols[0]
    
    # Create binary groups if attrition exists
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    df.boxplot(column=sat_col, by='attrition_flag', ax=axes[0], grid=False)
    axes[0].set_title(f'{sat_col} by Attrition', fontsize=14, fontweight='bold')
    axes[0].set_ylabel(sat_col)
    axes[0].set_xlabel('Attrition (0=Stayed, 1=Left)')
    
    # Distribution
    for val in [0, 1]:
        data = df[df['attrition_flag'] == val][sat_col].dropna()
        axes[1].hist(data, bins=20, alpha=0.6, label=f'{"Left" if val == 1 else "Stayed"}', edgecolor='white')
    axes[1].set_title(f'{sat_col} Distribution by Attrition', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(sat_col)
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '7_satisfaction_vs_attrition.png'))
    plt.close()
    
    # Stats comparison
    stayed = df[df['attrition_flag'] == 0][sat_col].dropna()
    left = df[df['attrition_flag'] == 1][sat_col].dropna()
    
    print(f'\n--- {sat_col}: Stayed vs Left ---')
    print(f'Stayed (n={len(stayed)}): Mean={stayed.mean():.3f}, Median={stayed.median():.3f}, Std={stayed.std():.3f}')
    print(f'Left (n={len(left)}): Mean={left.mean():.3f}, Median={left.median():.3f}, Std={left.std():.3f}')
    
    # Statistical test
    if len(stayed) > 0 and len(left) > 0:
        t_stat, p_value = stats.ttest_ind(stayed, left, equal_var=False)
        print(f'\nWelch t-test: t={t_stat:.4f}, p={p_value:.4f}')
        print(f'Significant difference: {"YES" if p_value < 0.05 else "NO"}')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((stayed.std()**2 + left.std()**2) / 2)
        cohens_d = (stayed.mean() - left.mean()) / pooled_std if pooled_std > 0 else 0
        print(f"Cohen's d: {cohens_d:.3f} (Effect Size)")

elif sat_cols:
    print(f'Satisfaction columns found: {sat_cols}')
    print(f'Satisfaction Stats:\n{df[sat_cols[0]].describe()}')
else:
    print('No satisfaction columns found. Checking all numeric columns with limited values...')
    for col in df.select_dtypes(include=[np.number]).columns:
        if 3 <= df[col].nunique() <= 10:
            print(f'  Potential satisfaction: {col} (unique: {sorted(df[col].dropna().unique())})')

print('[STATUS] Section 7 complete: Satisfaction vs Resignation')
```

```python
# === SECTION 8: Correlation Analysis ===
print('='*60)
print('SECTION 8: Correlation & Feature Interaction')
print('='*60)

num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
if num_df.shape[1] >= 2:
    corr_matrix = num_df.corr()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '8_correlation_matrix.png'))
    plt.close()
    
    # Top correlations
    corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    print('\nTop 10 Positive Correlations:')
    print(corr_pairs.head(10))
    print('\nTop 10 Negative Correlations:')
    print(corr_pairs.drop_duplicates().tail(10))
    
    # Feature interactions with attrition
    if 'attrition_flag' in num_df.columns:
        print('\n--- Correlations with Attrition ---')
        att_corr = corr_matrix['attrition_flag'].drop('attrition_flag').sort_values(ascending=False)
        print(att_corr.to_string())

print('[STATUS] Section 8 complete: Correlation Analysis')
```

```python
# === SECTION 9: Time Series (if applicable) ===
print('='*60)
print('SECTION 9: Time Series Patterns (if available)')
print('='*60)

date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date', 'time', 'year', 'month', 'quarter', 'period', 'hire', 'start', 'join'])]
print(f'Date columns: {date_cols}')

if date_cols:
    date_col = date_cols[0]
    try:
        df['_date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
        valid_dates = df['_date_parsed'].dropna()
        print(f'Date range: {valid_dates.min()} to {valid_dates.max()}')
        
        # Monthly trends if we have attrition
        if 'attrition_flag' in df.columns:
            df['_year_month'] = df['_date_parsed'].dt.to_period('M')
            monthly_attrition = df.groupby('_year_month')['attrition_flag'].mean() * 100
            
            fig, ax = plt.subplots(figsize=(14, 5))
            monthly_attrition.plot(marker='o', ax=ax, color='steelblue', linewidth=2)
            ax.set_title('Monthly Attrition Rate Trend', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Attrition Rate (%)')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, '9_monthly_attrition_trend.png'))
            plt.close()
            
            # Trend decomposition if enough data
            if len(monthly_attrition) >= 14:
                try:
                    ts = monthly_attrition.values
                    decomposition = sm.tsa.seasonal_decompose(ts, model='additive', period=12)
                    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
                    axes[0].plot(ts, label='Original')
                    axes[0].set_title('Attrition Rate - Original', fontweight='bold')
                    axes[0].legend()
                    axes[1].plot(decomposition.trend, label='Trend', color='green')
                    axes[1].set_title('Trend', fontweight='bold')
                    axes[1].legend()
                    axes[2].plot(decomposition.seasonal, label='Seasonal', color='orange')
                    axes[2].set_title('Seasonal', fontweight='bold')
                    axes[2].legend()
                    axes[3].plot(decomposition.resid, label='Residual', color='red')
                    axes[3].set_title('Residual', fontweight='bold')
                    axes[3].legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_DIR, '9_time_series_decomposition.png'))
                    plt.close()
                    print('Time series decomposition completed')
                except:
                    print('Could not perform decomposition (need more data points)')
    
    except Exception as e:
        print(f'Date parsing error: {e}')

print('[STATUS] Section 9 complete: Time Series')
```

```python
# === SECTION 10: Geographic Insights (if applicable) ===
print('='*60)
print('SECTION 10: Geographic & Demographic Insights')
print('='*60)

geo_cols = [c for c in df.columns if any(k in c.lower() for k in ['city', 'state', 'region', 'country', 'location', 'branch', 'office', 'site'])]
demo_cols = [c for c in df.columns if any(k in c.lower() for k in ['age', 'gender', 'education', 'degree', 'marital', 'ethnicity', 'race'])]

print(f'Geographic columns: {geo_cols}')
print(f'Demographic columns: {demo_cols}')

# Age analysis
age_cols = [c for c in df.columns if 'age' in c.lower()]
if age_cols:
    age_col = age_cols[0]
    print(f'\nAge Statistics:')
    print(df[age_col].describe())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df[age_col], bins=25, color='steelblue', edgecolor='white')
    axes[0].axvline(df[age_col].mean(), color='red', ls='--', label=f'Mean: {df[age_col].mean():.1f}')
    axes[0].set_title('Age Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Age')
    axes[0].legend()
    
    if 'attrition_flag' in df.columns:
        age_bins = [20, 30, 40, 50, 60, 100]
        age_labels = ['20-30', '31-40', '41-50', '51-60', '60+']
        df['age_group'] = pd.cut(df[age_col], bins=age_bins, labels=age_labels)
        age_attrition = df.groupby('age_group', observed=False)['attrition_flag'].mean() * 100
        age_attrition.plot(kind='bar', ax=axes[1], color='coral', edgecolor='white')
        axes[1].set_title('Attrition Rate by Age Group', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Age Group')
        axes[1].set_ylabel('Attrition Rate (%)')
        plt.sca(axes[1])
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '10_age_analysis.png'))
    plt.close()

# Gender analysis if exists
gender_cols = [c for c in df.columns if 'gender' in c.lower()]
if gender_cols:
    gen_col = gender_cols[0]
    print(f'\nGender Distribution:')
    print(df[gen_col].value_counts())
    print(df[gen_col].value_counts(normalize=True) * 100)

print('[STATUS] Section 10 complete: Geographic & Demographic')
```

```python
# === SECTION 11: Outlier Analysis ===
print('='*60)
print('SECTION 11: Business Outlier Analysis')
print('='*60)

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    
    if len(outliers) > 0:
        pct = len(outliers) / len(df) * 100
        print(f'\n{col}: {len(outliers)} outliers ({pct:.1f}%)')
        print(f'  Range: [{lower:.2f}, {upper:.2f}]')
        print(f'  Outlier values: {outliers[col].describe()}')

print('[STATUS] Section 11 complete: Outlier Analysis')
```

```python
# === SECTION 12: Feature Interaction Deep Dive ===
print('='*60)
print('SECTION 12: Feature Interaction Analysis')
print('='*60)

# If we have satisfaction, performance, and overtime all together
if 'sat_col' in dir() and 'perf_col' in dir() and 'ot_col' in dir():
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Satisfaction vs Performance (colored by attrition)
        if 'attrition_flag' in df.columns:
            scatter = axes[0].scatter(df[perf_col], df[sat_col], 
                                     c=df['attrition_flag'], cmap='coolwarm', alpha=0.5
I'll perform a comprehensive QC check on all outputs from the UCI Online Retail project. Let me start by examining all the output files.

```python
import argparse, os, pandas as pd
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir

if not OUTPUT_DIR:
    OUTPUT_DIR = r'C:\Users\Amorntep\DATA-Agent\projects\2026-05-01_uci_online_retail\output\quinn'
if not INPUT_PATH:
    INPUT_PATH = r'C:\Users\Amorntep\DATA-Agent\projects\2026-05-01_uci_online_retail\output'

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'[STATUS] OUTPUT_DIR: {OUTPUT_DIR}')
print(f'[STATUS] INPUT_PATH: {INPUT_PATH}')

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Scan all output files recursively
# ──────────────────────────────────────────────────────────────────────
project_root = Path(INPUT_PATH).parent if INPUT_PATH.endswith('.md') else Path(INPUT_PATH)
output_dir_path = project_root / 'output'

all_files = []
for ext in ['*.csv', '*.md', '*.py', '*.png', '*.json']:
    all_files.extend(list(output_dir_path.rglob(ext)))

print(f'\n[STATUS] Found {len(all_files)} output files:')
for f in sorted(all_files):
    rel = f.relative_to(project_root)
    size = f.stat().st_size
    print(f'  {rel} ({size:,} bytes)')

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Load all agent outputs into dicts
# ──────────────────────────────────────────────────────────────────────
agent_outputs = {
    'dana': {'csv': None, 'report': None},
    'eddie': {'report': None},
    'finn': {'csv': None, 'report': None},
    'mo': {'report': None},
    'max': {'report': None},
    'iris': {'report': None},
    'vera': {'report': None},
    'rex': {'report': None},
}

for f in all_files:
    rel = str(f.relative_to(project_root)).lower()
    content = f.read_text(encoding='utf-8', errors='replace')
    
    # Dana
    if 'dana' in rel and f.suffix == '.csv' and agent_outputs['dana']['csv'] is None:
        agent_outputs['dana']['csv'] = pd.read_csv(f)
    elif 'dana' in rel and f.suffix == '.md':
        agent_outputs['dana']['report'] = content
    
    # Eddie
    if 'eddie' in rel and f.suffix == '.md':
        agent_outputs['eddie']['report'] = content
    
    # Finn
    if 'finn' in rel and f.suffix == '.csv' and 'engineered' in rel and agent_outputs['finn']['csv'] is None:
        agent_outputs['finn']['csv'] = pd.read_csv(f)
    elif 'finn' in rel and f.suffix == '.md':
        agent_outputs['finn']['report'] = content
    
    # Mo
    if 'mo' in rel and f.suffix == '.md':
        agent_outputs['mo']['report'] = content
    
    # Max
    if 'max' in rel and f.suffix == '.md':
        agent_outputs['max']['report'] = content
    
    # Iris
    if 'iris' in rel and f.suffix == '.md':
        agent_outputs['iris']['report'] = content
    
    # Vera
    if 'vera' in rel and f.suffix == '.md':
        agent_outputs['vera']['report'] = content
    
    # Rex
    if 'rex' in rel and f.suffix == '.md':
        agent_outputs['rex']['report'] = content

print('\n[STATUS] Loaded agent outputs:')
for agent, outputs in agent_outputs.items():
    statuses = []
    for k, v in outputs.items():
        if v is not None:
            if isinstance(v, pd.DataFrame):
                statuses.append(f'{k}: DataFrame({v.shape})')
            else:
                statuses.append(f'{k}: {len(str(v)):,} chars')
        else:
            statuses.append(f'{k}: None')
    print(f'  {agent}: {", ".join(statuses)}')

# ──────────────────────────────────────────────────────────────────────
# STEP 3: DATA QUALITY CHECK (Dana output)
# ──────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('STEP 3: DATA QUALITY CHECK')
print('='*60)

data_quality_issues = []

if agent_outputs['dana']['csv'] is not None:
    df_dana = agent_outputs['dana']['csv']
    print(f'[CHECK] Dana CSV shape: {df_dana.shape}')
    print(f'[CHECK] Columns: {df_dana.columns.tolist()}')
    print(f'[CHECK] Memory: {df_dana.memory_usage(deep=True).sum() / 1e6:.2f} MB')
    
    # Check for missing values
    missing = df_dana.isnull().sum()
    missing_cols = missing[missing > 0]
    print(f'[CHECK] Missing values: {missing_cols.to_dict() if len(missing_cols) > 0 else "None"}')
    
    # Check for duplicates
    dup_count = df_dana.duplicated().sum()
    print(f'[CHECK] Duplicates: {dup_count}')
    if dup_count > 1000:
        data_quality_issues.append(f'High duplicate count: {dup_count}')
    
    # Check for data types
    dtypes = df_dana.dtypes.value_counts()
    print(f'[CHECK] Data types: {dtypes.to_dict()}')
    
    # Basic stats for numeric columns
    numeric_cols = df_dana.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats = df_dana[numeric_cols].describe()
        print(f'[CHECK] Numeric columns stats (sample):')
        print(stats.iloc[:, :min(5, len(numeric_cols))].to_string())
    
    # Check for negative values in Quantity/Price
    if 'Quantity' in df_dana.columns:
        neg_qty = (df_dana['Quantity'] < 0).sum()
        if neg_qty > 0:
            data_quality_issues.append(f'NEGATIVE Quantity values: {neg_qty}')
            print(f'[WARN] Negative Quantity: {neg_qty}')
    
    if 'UnitPrice' in df_dana.columns:
        neg_price = (df_dana['UnitPrice'] < 0).sum()
        if neg_price > 0:
            data_quality_issues.append(f'NEGATIVE UnitPrice values: {neg_price}')
            print(f'[WARN] Negative UnitPrice: {neg_price}')
    
    # Check if CustomerID has substantial missing
    if 'CustomerID' in df_dana.columns:
        missing_cust = df_dana['CustomerID'].isnull().sum()
        cust_missing_pct = missing_cust / len(df_dana) * 100
        if cust_missing_pct > 20:
            data_quality_issues.append(f'CustomerID missing: {cust_missing_pct:.1f}%')
            print(f'[WARN] CustomerID missing: {cust_missing_pct:.1f}%')
else:
    data_quality_issues.append('Dana CSV not found')
    print('[WARN] Dana CSV not loaded')

# ──────────────────────────────────────────────────────────────────────
# STEP 4: FEATURE ENGINEERING CHECK (Finn)
# ──────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('STEP 4: FEATURE ENGINEERING CHECK')
print('='*60)

feature_issues = []
engineered_cols = []

if agent_outputs['finn']['csv'] is not None:
    df_finn = agent_outputs['finn']['csv']
    print(f'[CHECK] Finn CSV shape: {df_finn.shape}')
    print(f'[CHECK] Columns ({len(df_finn.columns)}): {df_finn.columns.tolist()}')
    
    # Check for NaN/Inf in engineered features
    nan_counts = df_finn.isnull().sum()
    nan_features = nan_counts[nan_counts > 0]
    if len(nan_features) > 0:
        feature_issues.append(f'NaN values in features: {nan_features.to_dict()}')
        print(f'[WARN] NaN in features: {nan_features.to_dict()}')
    
    # Check for infinity values
    inf_counts = np.isinf(df_finn.select_dtypes(include=[np.number])).sum().sum()
    if inf_counts > 0:
        feature_issues.append(f'Infinity values found: {inf_counts}')
        print(f'[WARN] Infinity values: {inf_counts}')
    
    # Check for features with zero variance
    for col in df_finn.select_dtypes(include=[np.number]).columns:
        if df_finn[col].std() == 0:
            feature_issues.append(f'Zero variance feature: {col}')
            print(f'[WARN] Zero variance: {col}')
    
    engineered_cols = [c for c in df_finn.columns if c not in ['CustomerID']]
else:
    feature_issues.append('Finn engineered CSV not found')

if agent_outputs['finn']['report'] is not None:
    finn_report = agent_outputs['finn']['report']
    # Check for feature names in report
    if 'feature' in finn_report.lower() or 'engineer' in finn_report.lower():
        print('[CHECK] Finn report mentions features/engineering')
    if 'leakage' in finn_report.lower():
        print('[CHECK] Finn report mentions leakage')
    if 'scaling' in finn_report.lower() or 'normaliz' in finn_report.lower():
        print('[CHECK] Finn report mentions scaling/normalization')

# ──────────────────────────────────────────────────────────────────────
# STEP 5: MODEL PERFORMANCE CHECK — Parse all model results from Mo & Max
# ──────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('STEP 5: MODEL PERFORMANCE CHECK')
print('='*60)

model_issues = []
model_metrics = {}

# Parse Mo report
if agent_outputs['mo']['report'] is not None:
    mo_report = agent_outputs['mo']['report']
    
    # Extract metrics using pattern matching
    import re
    
    # Look for Churn model metrics
    churn_metrics = {}
    churn_section = re.search(r'Churn[^:]*:?\s*\n(.*?)(?=\n\n|\Z)', mo_report, re.DOTALL | re.IGNORECASE)
    
    # Extract numbers: F1, AUC, Accuracy, Precision, Recall
    f1_scores = re.findall(r'f1[^:]*:\s*([\d.]+)', mo_report, re.IGNORECASE)
    auc_scores = re.findall(r'auc[^:]*:\s*([\d.]+)', mo_report, re.IGNORECASE)
    accuracy_scores = re.findall(r'accuracy[^:]*:\s*([\d.]+)', mo_report, re.IGNORECASE)
    precision_scores = re.findall(r'precision[^:]*:\s*([\d.]+)', mo_report, re.IGNORECASE)
    recall_scores = re.findall(r'recall[^:]*:\s*([\d.]+)', mo_report, re.IGNORECASE)
    rmse_scores = re.findall(r'rmse[^:]*:\s*([\d.]+)', mo_report, re.IGNORECASE)
    mae_scores = re.findall(r'mae[^:]*:\s*([\d.]+)', mo_report, re.IGNORECASE)
    
    print(f'[CHECK] Mo report length: {len(mo_report):,} chars')
    print(f'[CHECK] F1 scores found: {f1_scores}')
    print(f'[CHECK] AUC scores found: {auc_scores}')
    print(f'[CHECK] Accuracy found: {accuracy_scores}')
    print(f'[CHECK] Precision found: {precision_scores}')
    print(f'[CHECK] Recall found: {recall_scores}')
    print(f'[CHECK] RMSE found: {rmse_scores}')
    print(f'[CHECK] MAE found: {mae_scores}')
    
    # Check for model comparison table
    has_comparison = 'comparison' in mo_report.lower() or 'compare' in mo_report.lower() or '|' in mo_report
    print(f'[CHECK] Has model comparison: {has_comparison}')
    
    # Check for overfitting discussion
    has_overfitting = 'overfit' in mo_report.lower() or 'train' in mo_report.lower() and 'test' in mo_report.lower() and 'gap' in mo_report.lower()
    print(f'[CHECK] Has overfitting analysis: {has_overfitting}')
    
    # Check for cross-validation
    has_cv = 'cross' in mo_report.lower() or 'cv' in mo_report.lower()
    print(f'[CHECK] Has cross-validation: {has_cv}')
    
    # Check for feature importance
    has_feature_importance = 'importance' in mo_report.lower() or 'shap' in mo_report.lower()
    print(f'[CHECK] Has feature importance: {has_feature_importance}')
    
    # Check for calibration info
    has_calibration = 'calibr' in mo_report.lower() or 'brier' in mo_report.lower() or 'probability' in mo_report.lower()
    print(f'[CHECK] Has calibration: {has_calibration}')
    
    # Check for imbalance handling
    has_imbalance = 'imbal' in mo_report.lower() or 'smote' in mo_report.lower() or 'class_weight' in mo_report.lower()
    print(f'[CHECK] Has imbalance handling: {has_imbalance}')
    
    # Check for threshold economics
    has_threshold_economics = 'threshold' in mo_report.lower() and ('cost' in mo_report.lower() or 'value' in mo_report.lower() or 'benefit' in mo_report.lower())
    print(f'[CHECK] Has threshold economics: {has_threshold_economics}')
    
    # Check for PR-AUC
    has_prauc = 'pr-auc' in mo_report.lower() or 'pr_auc' in mo_report.lower() or 'average precision' in mo_report.lower()
    print(f'[CHECK] Has PR-AUC: {has_prauc}')
    
    # Check for time-based validation
    has_time_split = 'time' in mo_report.lower() and ('split' in mo_report.lower() or 'oot' in mo_report.lower() or 'out of' in mo_report.lower())
    print(f'[CHECK] Has time-based validation: {has_time_split}')
    
    # Check for CLV model
    has_clv = 'clv' in mo_report.lower() or 'ltv' in mo_report.lower() or 'lifetime value' in mo_report.lower()
    print(f'[CHECK] Has CLV/LTV model: {has_clv}')
    
    # Check for inventory model
    has_inventory = 'inventory' in mo_report.lower() or 'stock' in mo_report.lower() or 'demand forecast' in mo_report.lower()
    print(f'[CHECK] Has inventory/stock model: {has_inventory}')
    
    # Benchmark dependencies
    has_xgboost = 'xgboost' in mo_report.lower() or 'xgb' in mo_report.lower()
    has_lightgbm = 'lightgbm' in mo_report.lower() or 'lgbm' in mo_report.lower()
    has_catboost = 'catboost' in mo_report.lower()
    print(f'[CHECK] XGBoost: {has_xgboost}, LightGBM: {has_lightgbm}, CatBoost: {has_catboost}')
    
    # Build model_metrics
    model_metrics = {
        'f1_scores': [float(s) for s in f1_scores if s.replace('.', '').isdigit()],
        'auc_scores': [float(s) for s in auc_scores if s.replace('.', '').isdigit()],
        'accuracy': [float(s) for s in accuracy_scores if s.replace('.', '').isdigit()],
        'has_comparison': has_comparison,
        'has_overfitting_analysis': has_overfitting,
        'has_cv': has_cv,
        'has_feature_importance': has_feature_importance,
        'has_calibration': has_calibration,
        'has_imbalance_handling': has_imbalance,
        'has_threshold_economics': has_threshold_economics,
        'has_prauc': has_prauc,
        'has_time_split': has_time_split,
        'has_clv_model': has_clv,
        'has_inventory_model': has_inventory,
        'benchmarked_xgboost': has_xgboost,
        'benchmarked_lightgbm': has_lightgbm,
        'benchmarked_catboost': has_catboost,
    }
    
    # Evaluate metrics against thresholds
    if f1_scores:
        for s in f1_scores:
            try:
                val = float(s)
                if val < 0.7:
                    model_issues.append(f'Low F1 score: {val}')
            except:
                pass
    if auc_scores:
        for s in auc_scores:
            try:
                val = float(s)
                if val < 0.7:
                    model_issues.append(f'Low AUC: {val}')
            except:
                pass
else:
    model_issues.append('Mo report not found')
    print('[WARN] Mo report not loaded')

# Parse Max report
if agent_outputs['max']['report'] is not None:
    max_report = agent_outputs['max']['report']
    print(f'\n[CHECK] Max report length: {len(max_report):,} chars')
    
    has_inventory_section = 'inventory' in max_report.lower() or 'stock' in max_report.lower() or 'reorder' in max_report.lower()
    has_optimization = 'optim' in max_report.lower() or 'pareto' in max_report.lower() or 'abc' in max_report.lower()
    has_support = 'support' in max_report.lower() or 'lift' in max_report.lower() or 'confidence' in max_report.lower()
    
    print(f'[CHECK] Has inventory optimization: {has_inventory_section}')
    print(f'[CHECK] Has optimization technique: {has_optimization}')
    print(f'[CHECK] Has pattern validation: {has_support}')
    
    model_metrics.update({
        'max_has_inventory': has_inventory_section,
        'max_has_optimization': has_optimization,
        'max_has_pattern_validation': has_support,
    })
else:
    model_issues.append('Max report not found')

# ──────────────────────────────────────────────────────────────────────
# STEP 6: BUSINESS READINESS CHECK (Iris + Rex)
# ──────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('STEP 6: BUSINESS READINESS CHECK')
print('='*60)

business_issues = []
business_scores = {}

# Iris check
if agent_outputs['iris']['report'] is not None:
    iris_report = agent_outputs['iris']['report']
    print(f'[CHECK] Iris report length: {len(iris_report):,} chars')
    
    has_action_items = 'action' in iris_report.lower() or 'recommend' in iris_report.lower()
    has_business_impact = 'impact' in iris_report.lower() or 'revenue' in iris_report.lower() or 'cost' in iris_report.lower()
    has_so_what = 'so what' in iris_report.lower() or 'therefore' in iris_report.lower() or 'key takeaway' in iris_report.lower()
    has_kpi = 'kpi' in iris_report.lower() or 'metric' in iris_report.lower() or 'roi' in iris_report.lower()
    
    print(f'[CHECK] Has action items: {has_action_items}')
    print(f'[CHECK] Has business impact: {has_business_impact}')
    print(f'[CHECK] Has "so what" framing: {has_so_what}')
    print(f'[CHECK] Has KPIs: {has_kpi}')
    
    # Count actionable insights (sections with recommendations)
    action_sections = len(re.findall(r'##+\s*(?:recommend|action|insight|takeaway)', iris_report, re.IGNORECASE))
    print(f'[CHECK] Action sections found: {action_sections}')
    
    if not has_action_items:
        business_issues.append('No actionable recommendations in Iris report')
    if not has_business_impact:
        business_issues.append('No business impact quantification')
    if not has_so_what:
        business_issues.append('Missing "so what" framing')
    
    # Count recommendations
    insight_count = len(re.findall(r'(?:^|\n)(?:\d+[.)]|\*|-)\s*(?:[A-Z][^.]*\b(recommend|action|implement|invest|focus|target|increase|reduce|optimize)\b[^.]*\.)', iris_report, re.MULTILINE | re.IGNORECASE))
    print(f'[CHECK] Specific recommendations found: {insight_count}')
    
    business_scores['iris_action_items'] = action_sections
    business_scores['iris_has_business_impact'] = has_business_impact
    business_scores['iris_insight_count'] = insight_count
else:
    business_issues.append('Iris report not found')

# Rex check
if agent_outputs['rex']['report'] is not None:
    rex_report = agent_outputs['rex']['report']
    print(f'\n[CHECK] Rex report length: {len(rex_report):,} chars')
    
    has_exec_summary = 'execut' in rex_report.lower() or 'summary' in rex_report.lower()
    has_limitations = 'limitation' in rex_report.lower() or 'assumption' in rex_report.lower() or 'risk' in rex_report.lower()
    has_next_steps = 'next step' in rex_report.lower() or 'recommendation' in rex_report.lower() or 'roadmap' in rex_report.lower()
    has_production = 'production' in rex_report.lower() or 'deploy' in rex_report.lower() or 'monitoring' in rex_report.lower()
    
    print(f'[CHECK] Has executive summary: {has_exec_summary}')
    print(f'[CHECK] Has limitations/risks: {has_limitations}')
    print(f'[CHECK] Has next steps: {has_next_steps}')
    print(f'[CHECK] Has production readiness: {has_production}')
    
    if not has_exec_summary:
        business_issues.append('Missing executive summary in Rex report')
    if not has_limitations:
        business_issues.append('Missing limitations or assumptions')
    if not has_next_steps:
        business_issues.append('Missing next steps/recommendations')
    
    business_scores['rex_has_summary'] = has_exec_summary
    business_scores['rex_has_limitations'] = has_limitations
    business_scores['rex_has_next_steps'] = has_next_steps
    business_scores['rex_has_production'] = has_production
else:
    business_issues.append('Rex report not found')

# ──────────────────────────────────────────────────────────────────────
# STEP 7: DATA LEAKAGE DETECTION (critical check)
# ──────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('STEP 7: DATA LEAKAGE DETECTION')
print('='*60)

leakage_issues = []

if agent_outputs['finn']['csv'] is not None and agent_outputs['dana']['csv'] is not None:
    df_leak = agent_outputs['finn']['csv']
    
    # Check for target-like features (features that look like targets)
    target_like_keywords = ['churn', 'clv', 'ltv', 'customer_value', 'is_churn', 'label', 'target']
    for col in df_leak.columns:
        col_lower = col.lower()
        for keyword in target_like_keywords:
            if keyword in col_lower:
                # Check if this is actually a feature (might be target leakage)
                if col not in ['CustomerID', 'InvoiceNo', 'StockCode']:
                    if col != 'is_churn' and col != 'CLV':  # These ARE targets, not leakage
                        # But check if target appears in features incorrectly
                        print(f'[CHECK] Feature "{col}" contains keyword "{keyword}" - verify no leakage')
    
    # Check for future-looking features (date-based)
    date_cols = [c for c in df_leak.columns if 'date' in c.lower() or 'time' in c.lower() or 'year' in c.lower() or 'month' in c.lower() or 'day' in c.lower()]
    if len(date_cols) > 0:
        print(f'[CHECK] Date/time features found: {date_cols}')
    
    # Check for high correlation between features (multi-collinearity)
    numeric_df = df_leak.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [(col1, col2, corr_matrix.iloc[i, j]) 
                     for i, col1 in enumerate(corr_matrix.columns) 
                     for j, col2 in enumerate(corr_matrix.columns) 
                     if i < j and corr_matrix.iloc[i, j] > 0.95]
        if high_corr:
            for col1, col2, corr_val in high_corr[:10]:
                leak_msg = f'High correlation (>0.95) between {col1} and {col2}: {corr_val:.3f}'
                leakage_issues.append(leak_msg)
                print(f'[WARN] {leak_msg}')
        else:
            print('[CHECK] No high-correlation pairs found')
    
    # KS test for distribution drift between first/last 20% of data (simulated)
    if len(df_leak) > 100:
        from scipy import stats
        
        try:
            split_point = int(len(df_leak) * 0.8)
            drift_cols = []
            for col in numeric_df.columns[:min(10, len(numeric_df.columns))]:
                early_data = numeric_df[col].iloc[:split_point].dropna()
                late_data = numeric_df[col].iloc[split_point:].dropna()
                if len(early_data) > 30 and len(late_data) > 30:
                    try:
                        stat, p = stats.ks_2samp(early_data, late_data)
                        if p < 0.05:
                            drift_cols.append((col, p))
                    except:
                        pass
            
            if drift_cols:
                print(f'[WARN] Distribution drift detected in {len(drift_cols)} features: {drift_cols[:5]}')
                leakage_issues.append(f'Distribution drift in {len(drift_cols)} features')
            else:
                print('[CHECK] No significant distribution drift detected')
        except ImportError:
            print('[CHECK] scipy.stats not available - skipping KS test')
        except Exception as e:
            print(f'[CHECK] KS test error (non-critical): {e}')
else:
    leakage_issues.append('Cannot perform leakage check - missing data')

# ──────────────────────────────────────────────────────────────────────
# STEP 8: VERIFICATION CHECK (Vera visuals)
# ──────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('STEP 8: VERIFICATION CHECK')
print('='*60)

visual_issues = []

if agent_outputs['vera']['report'] is not None:
    vera_report = agent_outputs['vera']['report']
    print(f'[CHECK] Vera report length: {len(vera_report):,} chars')
    
    has_charts = '.png' in vera_report or '.jpg' in vera_report or '.svg' in vera_report or 'chart' in vera_report.lower() or 'plot' in vera_report.lower()
    has_caveats = 'caveat' in vera_report.lower() or 'note' in vera_report.lower() or 'caution' in vera_report.lower()
    
    print(f'[CHECK] Has charts/plots: {has_charts}')
    print(f'[CHECK] Has caveats: {has_caveats}')
    
    if not has_charts:
        visual_issues.append('No visualizations mentioned in Vera report')
    
    # Check for visual files
    png_files = [f for f in all_files if f.suffix == '.png']
    print(f'[CHECK] Image files in output: {len(png_files)}')
    
    # Check for misleading visualizations
    if 'misleading' in vera_report.lower():
        visual_issues.append('Vera mentions misleading visuals')
else:
    visual_issues.append('Vera report not found')

# ──────────────────────────────────────────────────────────────────────
# STEP 9: COMPILE FINAL QC RESULTS
# ──────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('STEP 9: COMPILING QC RESULTS')
print('='*60)

# Collect all issues
all_issues = data_quality_issues + feature_issues + model_issues + leakage_issues + business_issues + visual_issues

# Determine QC verdict categories
tech_issues = data_quality_issues + feature_issues + leakage_issues
model_check_issues = model_issues
business_issues = business_issues
visual_issues = visual_issues

# Score each category (0 = fail, 1 = pass)
data_quality_pass = len(data_quality_issues) == 0
feature_eng_pass = len(feature_issues) == 0
leakage_pass = len(leakage_issues) == 0
model_perf_pass = len(model_issues) == 0
business_pass = len(business_issues) == 0
visual_pass = len(visual_issues) == 0

# Check WORLD_CLASS_QC criteria
# 1. Imbalance metrics
imbalance_pass = model_metrics.get('has_imbalance_handling', False) and model_metrics.get('has_prauc', False)
# 2. Validation realism  
validation_pass = model_metrics.get('has_time_split', False) or model_metrics.get('has_cv', False)
# 3. Threshold economics
threshold_pass = model_metrics.get('has_threshold_economics', False)
# 4. Calibration
calibration_pass = model_metrics.get('has_calibration', True)  # Regression may not have calibration
# 5. Tabular benchmark
tabular_benchmark = model_metrics.get('benchmarked_xgboost', False) or model_metrics.get('benchmarked_lightgbm', False) or model_metrics.get('benchmarked_catboost', False)

print(f'\n[QC SCORE] Data Quality: {"PASS" if data_quality_pass else "FAIL"} ({len(data_quality_issues)} issues)')
print(f'[QC SCORE] Feature Engineering: {"PASS" if feature_eng_pass else "FAIL"} ({len(feature_issues)} issues)')
print(f'[QC SCORE] Data Leakage: {"PASS" if leakage_pass else "FAIL"} ({len(leakage_issues)} issues)')
print(f'[QC SCORE] Model Performance: {"PASS" if model_perf_pass else "FAIL"} ({len(model_issues)} issues)')
print(f'[QC SCORE] Business Readiness: {"PASS" if business_pass else "FAIL"} ({len(business_issues)} issues)')
print(f'[QC SCORE] Visual Quality: {"PASS" if visual_pass else "FAIL"} ({len(visual_issues)} issues)')
print(f'\n[WORLD_CLASS QC] Imbalance: {"PASS" if imbalance_pass else "FAIL"}')
print(f'[WORLD_CLASS QC] Validation: {"PASS" if validation_pass else "FAIL"}')
print(f'[WORLD_CLASS QC] Threshold Economics: {"PASS" if threshold_pass else "FAIL"}')
print(f'[WORLD_CLASS QC] Calibration: {"PASS" if calibration_pass else "FAIL"}')
print(f'[WORLD_CLASS QC] Tabular Benchmark: {"PASS" if tabular_benchmark else "FAIL"}')

# ──────────────────────────────────────────────────────────────────────
# STEP 10: DETERMINE VERDICT
# ──────────────────────────────────────────────────────────────────────
total_categories = 6  # Data, Features, Leakage, Model, Business, Visual
passed_categories = sum([data_quality_pass, feature_eng_pass, leakage_pass, model_perf_pass, business_pass, visual_pass])

# Business satisfaction criteria
biz_satisfaction_score = 0
biz_satisfaction_notes = []

# Criteria 1: Model performance ≥ threshold
# Assuming threshold of 0.70 for F1/AUC
has_good_metrics = model_metrics.get('f1_scores', []) and max(model_metrics['f1_scores']) >= 0.7
if has_good_metrics:
    biz_satisfaction_score += 1
    biz_satisfaction_notes.append('Model performance OK')
else:
    biz_satisfaction_notes.append('Model performance below threshold')

# Criteria 2: Actionable insights ≥ 2
insight_count = business_scores.get('iris_insight_count', 0)
has_actionable = business_scores.get('iris_action_items', 0) >= 2 or insight_count >= 2
if has_actionable:
    biz_satisfaction_score += 1
    biz_satisfaction_notes.append('Actionable insights OK')
else:
    biz_satisfaction_notes.append('Not enough actionable insights')

# Criteria 3: Business questions answered ≥ 80%
# Based on completeness of coverage
has_churn = 'churn' in str(agent_outputs.get('mo', {}).get('report', '')).lower() or 'churn' in str(agent_outputs.get('max', {}).get('report', '')).lower()
has_clv = model_metrics.get('has_clv_model', False)
has_inventory = model_metrics.get('has_inventory_model', False) or model_metrics.get('max_has_inventory', False)
has_recs = business_scores.get('iris_has_business_impact', False)

coverage_score = sum([has_churn, has_clv, has_inventory, has_recs]) / 4
print(f'\n[CHECK] Business coverage: Churn={has_churn}, CLV={has_clv}, Inventory={has_inventory}, Recommendations={has_recs}')
print(f'[CHECK] Coverage score: {coverage_score:.0%}')

if coverage_score >= 0.75:
    biz_satisfaction_score += 1
    biz_satisfaction_notes.append(f'Business coverage OK ({coverage_score:.0%})')
else:
    biz_satisfaction_notes.append(f'Business coverage insufficient ({coverage_score:.0%})')

# Criteria 4: Technical soundness
tech_sound = data_quality_pass and feature_eng_pass and leakage_pass
if tech_sound:
    biz_satisfaction_score += 1
    biz_satisfaction_notes.append('Technical soundness OK')
else:
    biz_satisfaction_notes.append('Technical issues found')

print(f'\n[BUSINESS SATISFACTION] Score: {biz_satisfaction_score}/4')

# Determine verdict
restart_cycle = biz_satisfaction_score < 3

# Determine restart origin
if restart_cycle:
    restart_reasons = []
    if not data_quality_pass or not feature_eng_pass:
        restart_from = 'dana' if not data_quality_pass else 'finn'
        restart_reasons.append(f'Data/Feature issues -> restart from {restart_from}')
    elif not model_perf_pass:
        restart_from = 'mo'

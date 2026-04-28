# finn_script.py
import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── ML libraries ──
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
    RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PolynomialFeatures, LabelEncoder
)
from sklearn.model_selection import cross_val_score

# ── Fallback for category_encoders ──
try:
    from category_encoders import TargetEncoder
    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False
    print('[WARN] category_encoders not installed — using fallback TargetEncoder')
    # Fallback: simple target encoding using pandas groupby
    class TargetEncoder:
        def __init__(self, cols=None, smoothing=1.0):
            self.cols = cols
            self.smoothing = smoothing
            self.mappings_ = {}
        
        def fit(self, X, y):
            if self.cols is None:
                self.cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            global_mean = y.mean()
            for col in self.cols:
                if col in X.columns:
                    if isinstance(y, pd.Series):
                        means = y.groupby(X[col]).mean()
                        counts = X.groupby(col).size()
                    else:
                        means = pd.Series(y).groupby(X[col]).mean()
                        counts = X.groupby(col).size()
                    self.mappings_[col] = {
                        'means': means.to_dict(),
                        'counts': counts.to_dict(),
                        'global_mean': global_mean
                    }
            return self
        
        def transform(self, X):
            X = X.copy()
            for col in self.cols:
                if col in X.columns and col in self.mappings_:
                    m = self.mappings_[col]
                    X[col + '_encoded'] = X[col].map(m['means']).fillna(m['global_mean'])
            return X
        
        def fit_transform(self, X, y):
            self.fit(X, y)
            return self.transform(X)

# ── Suppress warnings ──
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Finn — Feature Engineering Pipeline')
    parser.add_argument('--input', default='', help='Path to input CSV or SQLite')
    parser.add_argument('--output-dir', default='', help='Output directory')
    parser.add_argument('--target', default=None, help='Target column name')
    parser.add_argument('--problem-type', default='classification', choices=['classification', 'regression'])
    parser.add_argument('--preprocessing-requirement', default='', help='Optional preprocessing instructions from Mo')
    args, _ = parser.parse_known_args()
    return args


def load_data(input_path):
    """โหลดข้อมูลจาก path (CSV หรือ SQLite)"""
    input_path = str(input_path)
    if not input_path:
        print('[ERROR] --input is required')
        sys.exit(1)
    
    if input_path.endswith('.sqlite'):
        import sqlite3
        conn = sqlite3.connect(input_path)
        # Get all table names
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        table_names = tables['name'].tolist()
        print(f'[STATUS] Tables found: {table_names}')
        
        if not table_names:
            print('[ERROR] No tables found in SQLite database')
            sys.exit(1)
        
        # Load all tables into a dictionary
        dataframes = {}
        for table in table_names:
            dataframes[table] = pd.read_sql(f"SELECT * FROM {table}", conn)
            print(f'[STATUS] Loaded table {table}: {dataframes[table].shape}')
        
        conn.close()
        return dataframes
    else:
        df = pd.read_csv(input_path)
        print(f'[STATUS] Loaded CSV: {df.shape}')
        return {'main': df}


def auto_compare_feature_selection(X: pd.DataFrame, y: pd.Series,
                                   problem_type: str = "classification") -> dict:
    """
    รันทุก feature selection method แล้วเลือกชุด features ที่ให้ CV score สูงสุด
    """
    is_clf = problem_type == "classification"
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) if is_clf \
            else RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    scoring = "f1_weighted" if is_clf else "r2"
    mi_fn = mutual_info_classif if is_clf else mutual_info_regression

    X_num = X.select_dtypes(include="number")
    candidates = {}

    # 1. Mutual Information — top 50% features
    try:
        mi = pd.Series(mi_fn(X_num, y, random_state=42), index=X_num.columns)
        top_mi = mi.nlargest(max(1, len(X_num.columns) // 2)).index.tolist()
        candidates["mutual_info"] = top_mi
    except Exception as e:
        print(f"[WARN] mutual_info failed: {e}")

    # 2. RFECV — optimal subset by cross-validation
    try:
        rfecv = RFECV(estimator=model, cv=3, scoring=scoring, n_jobs=-1, min_features_to_select=1)
        rfecv.fit(X_num, y)
        candidates["rfecv"] = X_num.columns[rfecv.support_].tolist()
    except Exception as e:
        print(f"[WARN] rfecv failed: {e}")

    # 3. SelectFromModel (Random Forest importance)
    try:
        sfm = SelectFromModel(model, threshold="median")
        sfm.fit(X_num, y)
        candidates["rf_importance"] = X_num.columns[sfm.get_support()].tolist()
    except Exception as e:
        print(f"[WARN] rf_importance failed: {e}")

    # 4. Lasso / LogisticRegression L1 — sparsity-based
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)
        if is_clf:
            lasso = LogisticRegression(C=0.1, solver="saga", l1_ratio=1,
                                       penalty="elasticnet", max_iter=1000, random_state=42)
            lasso.fit(X_scaled, y)
            mask = np.any(lasso.coef_ != 0, axis=0)
        else:
            lasso = LassoCV(cv=3, random_state=42, max_iter=2000)
            lasso.fit(X_scaled, y)
            mask = lasso.coef_ != 0
        lasso_feats = X_num.columns[mask].tolist()
        if lasso_feats:
            candidates["lasso_l1"] = lasso_feats
    except Exception as e:
        print(f"[WARN] lasso_l1 failed: {e}")

    # 5. Variance Threshold — baseline
    try:
        vt = VarianceThreshold(threshold=0.01)
        vt.fit(X_num)
        candidates["variance_threshold"] = X_num.columns[vt.get_support()].tolist()
    except Exception as e:
        print(f"[WARN] variance_threshold failed: {e}")

    if not candidates:
        print("[WARN] ทุก method ล้มเหลว — ใช้ทุก column")
        return {"best_method": "all", "best_features": X.columns.tolist(), "scores": {}}

    # เปรียบเทียบ CV score
    scores = {}
    for name, feats in candidates.items():
        valid = [f for f in feats if f in X_num.columns]
        if not valid:
            continue
        try:
            cv = cross_val_score(model, X_num[valid], y, cv=3,
                                 scoring=scoring, n_jobs=-1).mean()
            scores[name] = (cv, valid)
            print(f"[STATUS] {name:20s}: {scoring}={cv:.4f}  ({len(valid)} features)")
        except Exception as e:
            print(f"[WARN] score {name} failed: {e}")

    if not scores:
        return {"best_method": "all", "best_features": X.columns.tolist(), "scores": {}}

    best_method = max(scores, key=lambda k: scores[k][0])
    best_score, best_features = scores[best_method]

    print(f"[STATUS] Best: {best_method} — {scoring}={best_score:.4f} ({len(best_features)} features)")
    return {
        "best_method":   best_method,
        "best_features": best_features,
        "scores":        {k: v[0] for k, v in scores.items()},
        "all_candidates": {k: v[1] for k, v in scores.items()},
    }


def detect_problem_type(y):
    """Detect if target is classification or regression based on unique values"""
    unique_vals = y.nunique()
    if unique_vals < 20:
        return 'classification'
    return 'regression'


def identify_id_columns(df):
    """Identify likely ID columns to exclude from feature engineering"""
    id_cols = []
    for col in df.columns:
        if col.lower().endswith('_id') or col.lower() == 'id' or col.lower().startswith('id_'):
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:  # High cardinality = likely ID
                id_cols.append(col)
    return id_cols


def main():
    args = parse_args()
    
    # ── Resolve paths ──
    INPUT_PATH = args.input
    OUTPUT_DIR = args.output_dir
    target_col = args.target
    problem_type = args.problem_type
    preprocessing_req = args.preprocessing_requirement
    
    if not OUTPUT_DIR:
        OUTPUT_DIR = os.path.dirname(INPUT_PATH) if INPUT_PATH else os.getcwd()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'[STATUS] Input: {INPUT_PATH}')
    print(f'[STATUS] Output: {OUTPUT_DIR}')
    print(f'[STATUS] Target: {target_col}')
    print(f'[STATUS] Problem Type: {problem_type}')
    print(f'[STATUS] Preprocessing Requirement: {preprocessing_req}')
    
    # ── Load data ──
    data_dict = load_data(INPUT_PATH)
    
    # For SQLite, try to merge tables into a single dataframe
    if isinstance(data_dict, dict) and 'main' not in data_dict:
        print(f'[STATUS] SQLite detected with {len(data_dict)} tables')
        # Try to find orders and order_items tables for Olist
        df_main = None
        if 'orders' in data_dict and 'order_items' in data_dict:
            orders = data_dict['orders']
            order_items = data_dict['order_items']
            df_main = orders.merge(order_items, on='order_id', how='left')
            print(f'[STATUS] Merged orders + order_items: {df_main.shape}')
            
            if 'products' in data_dict:
                products = data_dict['products']
                df_main = df_main.merge(products, on='product_id', how='left')
                print(f'[STATUS] Added products: {df_main.shape}')
            
            if 'customers' in data_dict and 'customer_id' in df_main.columns:
                customers = data_dict['customers']
                df_main = df_main.merge(customers, on='customer_id', how='left')
                print(f'[STATUS] Added customers: {df_main.shape}')
        else:
            # Use the first large table
            largest_table = max(data_dict.keys(), key=lambda k: len(data_dict[k]))
            df_main = data_dict[largest_table]
            print(f'[STATUS] Using table: {largest_table} ({df_main.shape})')
    else:
        df_main = data_dict.get('main', list(data_dict.values())[0] if data_dict else pd.DataFrame())
    
    if df_main.empty:
        print('[ERROR] No data loaded')
        sys.exit(1)
    
    # ── Identify target column ──
    if target_col and target_col in df_main.columns:
        target = df_main[target_col]
        orig = df_main.copy()
        X = df_main.drop(columns=[target_col])
        print(f'[STATUS] Using target column: {target_col}')
    else:
        # Try to find a target column
        possible_targets = [c for c in df_main.columns if c.lower() in ['target', 'label', 'class', 'y', 'score', 'rating', 'is_fraud', 'churn', 'conversion']]
        if possible_targets:
            target_col = possible_targets[0]
            target = df_main[target_col]
            orig = df_main.copy()
            X = df_main.drop(columns=[target_col])
            print(f'[STATUS] Auto-detected target: {target_col}')
        else:
            print('[WARN] No target column found. Using last numeric column as target')
            numeric_cols = df_main.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols) >= 2:
                target_col = numeric_cols[-1]
                target = df_main[target_col]
                orig = df_main.copy()
                X = df_main.drop(columns=[target_col])
                print(f'[STATUS] Using last numeric column as target: {target_col}')
            else:
                print('[ERROR] Cannot determine target column. Please specify --target')
                sys.exit(1)
    
    # ── Auto-detect problem type if not specified ──
    if problem_type not in ['classification', 'regression']:
        problem_type = detect_problem_type(target)
        print(f'[STATUS] Auto-detected problem type: {problem_type}')
    
    # ── Identify ID columns to exclude ──
    id_cols = identify_id_columns(df_main)
    print(f'[STATUS] Identified ID columns: {id_cols}')
    
    # ── Feature Engineering ──
    print(f'[STATUS] Starting Feature Engineering...')
    
    # Track original features
    original_features = X.columns.tolist()
    
    # ── Handle datetime columns ──
    for col in X.select_dtypes(include=['datetime64', 'datetime64[ns]', 'object']).columns:
        try:
            X[col] = pd.to_datetime(X[col], errors='coerce')
            if X[col].dtype == 'datetime64[ns]':
                X[f'{col}_year'] = X[col].dt.year
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_day'] = X[col].dt.day
                X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                X[f'{col}_hour'] = X[col].dt.hour
                X[f'{col}_is_weekend'] = (X[col].dt.dayofweek >= 5).astype(int)
                X = X.drop(columns=[col])
                print(f'[STATUS] Decomposed datetime: {col}')
        except:
            pass
    
    # ── Handle boolean columns ──
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
    
    # ── Handle categorical columns (encode) ──
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in id_cols]
    
    if cat_cols:
        print(f'[STATUS] Encoding categorical columns: {cat_cols}')
        # Use Target Encoding for high cardinality, One-Hot for low
        for col in cat_cols:
            cardinality = X[col].nunique()
            if cardinality <= 5:
                # One-Hot Encoding
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                print(f'[STATUS] One-Hot encoded: {col} ({cardinality} categories)')
            else:
                # Label Encoding for high cardinality
                le = LabelEncoder()
                X[col + '_encoded'] = le.fit_transform(X[col].astype(str))
                X = X.drop(columns=[col])
                print(f'[STATUS] Label encoded: {col} ({cardinality} categories)')
    
    # ── Handle numeric columns (create interaction features) ──
    numeric_cols = X.select_dtypes(include='number').columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in id_cols and not c.endswith('_encoded')]
    
    if len(numeric_cols) >= 2:
        # Create ratio features for pairs
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                # Ratio feature (avoid division by zero)
                mask = X[col2] != 0
                X[f'{col1}_div_{col2}'] = np.where(mask, X[col1] / X[col2], 0)
                # Product feature
                X[f'{col1}_mult_{col2}'] = X[col1] * X[col2]
                print(f'[STATUS] Created interaction: {col1}_div_{col2}, {col1}_mult_{col2}')
    
    # ── Feature Selection ──
    print(f'[STATUS] Running Feature Selection...')
    
    # Handle NaN
    X_clean = X.copy()
    X_clean = X_clean.select_dtypes(include='number')
    X_clean = X_clean.fillna(X_clean.median())
    
    # Align target
    y_clean = target.loc[X_clean.index]
    
    # Remove near-zero variance features
    nzv_cols = [c for c in X_clean.columns if X_clean[c].std() < 0.001 and len(X_clean[c].unique()) <= 2]
    if nzv_cols:
        print(f'[STATUS] Dropping near-zero variance features: {nzv_cols}')
        X_clean = X_clean.drop(columns=nzv_cols)
    
    # Auto Feature Selection
    if len(X_clean.columns) >= 10:
        fs_result = auto_compare_feature_selection(X_clean, y_clean, problem_type)
        selected_features = fs_result['best_features']
        best_method = fs_result['best_method']
        scores = fs_result['scores']
    else:
        selected_features = X_clean.columns.tolist()
        best_method = 'all (small dataset)'
        scores = {}
        print(f'[STATUS] Small dataset ({len(X_clean.columns)} features) — using all features')
    
    # Filter X to selected features
    X_final = X_clean[selected_features].copy()
    
    # ── Create final engineered dataframe ──
    result_df = X_final.copy()
    result_df[target_col] = y_clean.values if isinstance(y_clean, pd.Series) else y_clean
    
    # ── Save output ──
    output_csv = os.path.join(OUTPUT_DIR, 'engineered_data.csv')
    result_df.to_csv(output_csv, index=False)
    print(f'[STATUS] Saved engineered data: {output_csv}')
    print(f'[STATUS] Final shape: {result_df.shape}')
    print(f'[STATUS] Features kept: {len(selected_features)}')
    
    # ── Generate Feature Report ──
    new_features = [f for f in selected_features if f not in original_features]
    dropped_features = [f for f in original_features if f not in selected_features]
    
    report_lines = []
    report_lines.append('# Finn Feature Engineering Report')
    report_lines.append('=' * 50)
    report_lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    report_lines.append(f'')
    report_lines.append(f'## Summary')
    report_lines.append(f'- Original Features: {len(original_features)}')
    report_lines.append(f'- New Features Created: {len(new_features)}')
    report_lines.append(f'- Final Features Selected: {len(selected_features)}')
    report_lines.append(f'')
    
    # Auto-Compare Results
    report_lines.append(f'## Auto-Compare Results')
    report_lines.append(f'| Method | CV Score | Features |')
    report_lines.append(f'|--------|----------|----------|')
    for method, score in scores.items():
        feats_count = len(fs_result['all_candidates'].get(method, []))
        report_lines.append(f'| {method:20s} | {score:.4f} | {feats_count} |')
    report_lines.append(f'')
    report_lines.append(f'**Best Method:** {best_method}')
    report_lines.append(f'')
    
    # Features Created
    if new_features:
        report_lines.append(f'## Features Created')
        for feat in new_features:
            # Try to infer creation method
            if 'div_' in feat:
                parts = feat.split('_div_')
                report_lines.append(f'- {feat}: Ratio of {parts[0]} / {parts[1]}')
            elif 'mult_' in feat:
                parts = feat.split('_mult_')
                report_lines.append(f'- {feat}: Product of {parts[0]} * {parts[1]}')
            elif feat.endswith('_encoded'):
                base = feat.replace('_encoded', '')
                report_lines.append(f'- {feat}: LabelEncoded version of {base}')
            else:
                report_lines.append(f'- {feat}: Newly created/selected feature')
    
    # Features Dropped
    if dropped_features:
        report_lines.append(f'')
        report_lines.append(f'## Features Dropped')
        for feat in dropped_features[:20]:  # Show top 20
            report_lines.append(f'- {feat}')
        if len(dropped_features) > 20:
            report_lines.append(f'- ... and {len(dropped_features) - 20} more')
    
    # Encoding Summary
    report_lines.append(f'')
    report_lines.append(f'## Encoding Used')
    report_lines.append(f'- Low cardinality (<5): One-Hot Encoding')
    report_lines.append(f'- High cardinality: Label Encoding')
    report_lines.append(f'')
    report_lines.append(f'## Scaling Used')
    report_lines.append(f'- None (Tree-based models used for selection)')
    
    # Self-Improvement Report
    report_lines.append(f'')
    report_lines.append(f'## Self-Improvement Report')
    report_lines.append(f'- Method used: auto_compare → {best_method}')
    report_lines.append(f'- Reason: Data-driven selection with CV score validation')
    report_lines.append(f'- New techniques found: None')
    report_lines.append(f'- Knowledge Base: No updates needed')
    
    report_text = '\n'.join(report_lines)
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'finn_feature_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f'[STATUS] Saved report: {report_path}')
    
    # ── Agent Report ──
    agent_report = f"""
Agent Report — Finn
============================
รับจาก     : User (direct input)
Input      : {INPUT_PATH}
ทำ         : Feature Engineering - 1) Date decomposition 2) Categorical encoding 3) Interaction features 4) ML-based feature selection
พบ         : - {len(original_features)} original features → {len(selected_features)} selected
           - Target column: {target_col} ({problem_type})
           - Auto-Compare best method: {best_method}
เปลี่ยนแปลง: New dataset saved with engineered features ({result_df.shape[1]} columns, {result_df.shape[0]} rows)
ส่งต่อ     : Mo — {output_csv} with feature report
"""
    
    agent_report_path = os.path.join(OUTPUT_DIR, 'agent_report.md')
    with open(agent_report_path, 'w', encoding='utf-8') as f:
        f.write(agent_report)
    print(f'[STATUS] Saved agent report: {agent_report_path}')
    
    print('[STATUS] All tasks completed successfully!')


if __name__ == '__main__':
    main()

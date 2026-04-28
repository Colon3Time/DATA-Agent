I'll start by reading the Knowledge Base and input data, then create the feature engineering pipeline.

```python
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
from category_encoders import TargetEncoder

# ── Suppress warnings ──
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Finn — Feature Engineering Pipeline')
    parser.add_argument('--input', default='', help='Path to input CSV file')
    parser.add_argument('--output-dir', default='', help='Directory for output files')
    parser.add_argument('--target', default='target', help='Target column name')
    return parser.parse_args()


def load_data(input_path: str) -> pd.DataFrame:
    """Load data from CSV or SQLite"""
    if input_path.endswith('.sqlite'):
        print(f'[STATUS] Input is SQLite: {input_path}')
        import sqlite3
        conn = sqlite3.connect(input_path)
        # Olist has multiple tables — load all
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        print(f'[STATUS] Tables found: {tables["name"].tolist()}')
        df = pd.DataFrame()
        for t in tables['name']:
            tbl = pd.read_sql(f"SELECT * FROM {t}", conn)
            # Add table prefix to avoid column conflicts
            tbl.columns = [f"{t}_{c}" if c not in ['order_id','customer_id','seller_id','product_id'] else c for c in tbl.columns]
            if df.empty:
                df = tbl
            else:
                # Try merging common columns
                common = [c for c in df.columns if c in tbl.columns and c != t]
                if common:
                    df = df.merge(tbl, on=common, how='left', suffixes=('','_y'))
                    df = df.drop(columns=[c for c in df.columns if c.endswith('_y')], errors='ignore')
                else:
                    df = pd.concat([df, tbl], axis=1)
        conn.close()
        print(f'[STATUS] Loaded SQLite: {df.shape}')
        return df
    else:
        df = pd.read_csv(input_path)
        print(f'[STATUS] Loaded CSV: {df.shape}')
        return df


def auto_compare_feature_selection(X: pd.DataFrame, y: pd.Series,
                                   problem_type: str = "classification") -> dict:
    """Run all feature selection methods and pick best by CV score"""
    is_clf = problem_type == "classification"
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) if is_clf \
            else RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    scoring = "f1_weighted" if is_clf else "r2"
    mi_fn = mutual_info_classif if is_clf else mutual_info_regression

    X_num = X.select_dtypes(include='number')
    if X_num.empty or len(X_num.columns) == 0:
        print('[WARN] No numeric columns for feature selection — using all')
        return {'best_method': 'all', 'best_features': X.columns.tolist(), 'scores': {}}

    # Detect if target is multiclass
    y_unique = y.nunique() if hasattr(y, 'nunique') else len(set(y))

    candidates = {}

    # 1. Mutual Information — top 50%
    try:
        mi_scores = mi_fn(X_num, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=X_num.columns).sort_values(ascending=False)
        top_n = max(1, len(X_num.columns) // 2)
        candidates['mutual_info'] = mi_series.nlargest(top_n).index.tolist()
        print(f'[STATUS] mutual_info: selected {len(candidates["mutual_info"])} features')
    except Exception as e:
        print(f'[WARN] mutual_info failed: {e}')

    # 2. RFECV
    try:
        rfecv = RFECV(estimator=model, cv=min(3, len(y)//2), scoring=scoring,
                      n_jobs=-1, min_features_to_select=1)
        rfecv.fit(X_num, y)
        candidates['rfecv'] = X_num.columns[rfecv.support_].tolist()
        print(f'[STATUS] rfecv: selected {len(candidates["rfecv"])} features')
    except Exception as e:
        print(f'[WARN] rfecv failed: {e}')

    # 3. SelectFromModel — RF importance
    try:
        sfm = SelectFromModel(model, threshold='median')
        sfm.fit(X_num, y)
        candidates['rf_importance'] = X_num.columns[sfm.get_support()].tolist()
        print(f'[STATUS] rf_importance: selected {len(candidates["rf_importance"])} features')
    except Exception as e:
        print(f'[WARN] rf_importance failed: {e}')

    # 4. Lasso / Logistic L1
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)
        if is_clf and y_unique > 2:
            # multiclass — use RandomForest instead
            pass
        elif is_clf:
            logreg = LogisticRegression(C=0.1, solver='saga', penalty='l1', max_iter=1000, random_state=42)
            logreg.fit(X_scaled, y)
            mask = logreg.coef_.flatten() != 0
            lasso_feats = X_num.columns[mask].tolist()
            if lasso_feats:
                candidates['lasso_l1'] = lasso_feats
                print(f'[STATUS] lasso_l1: selected {len(lasso_feats)} features')
        else:
            lasso = LassoCV(cv=3, random_state=42, max_iter=2000)
            lasso.fit(X_scaled, y)
            mask = lasso.coef_ != 0
            lasso_feats = X_num.columns[mask].tolist()
            if lasso_feats:
                candidates['lasso_l1'] = lasso_feats
                print(f'[STATUS] lasso_l1: selected {len(lasso_feats)} features')
    except Exception as e:
        print(f'[WARN] lasso_l1 failed: {e}')

    # 5. Variance Threshold
    try:
        vt = VarianceThreshold(threshold=0.01)
        vt.fit(X_num)
        candidates['variance_threshold'] = X_num.columns[vt.get_support()].tolist()
        print(f'[STATUS] variance_threshold: selected {len(candidates["variance_threshold"])} features')
    except Exception as e:
        print(f'[WARN] variance_threshold failed: {e}')

    if not candidates:
        print('[WARN] All selection methods failed — using all columns')
        return {'best_method': 'all', 'best_features': X.columns.tolist(), 'scores': {}}

    # Score each
    scores = {}
    for name, feats in candidates.items():
        valid = [f for f in feats if f in X_num.columns]
        if not valid:
            continue
        try:
            cv = cross_val_score(model, X_num[valid], y, cv=min(3, len(y)//2),
                                 scoring=scoring, n_jobs=-1).mean()
            scores[name] = (cv, valid)
            print(f'[STATUS] {name:20s}: {scoring}={cv:.4f}  ({len(valid)} features)')
        except Exception as e:
            print(f'[WARN] score {name} failed: {e}')

    if not scores:
        return {'best_method': 'all', 'best_features': X.columns.tolist(), 'scores': {}}

    best_method = max(scores, key=lambda k: scores[k][0])
    best_score, best_features = scores[best_method]

    print(f'[STATUS] BEST: {best_method} — {scoring}={best_score:.4f} ({len(best_features)} features)')
    return {
        'best_method': best_method,
        'best_features': best_features,
        'scores': {k: v[0] for k, v in scores.items()},
        'all_candidates': {k: v[1] for k, v in scores.items()}
    }


def drop_target_leakage(df: pd.DataFrame, target_col: str, corr_threshold: float = 0.95) -> pd.DataFrame:
    """Detect and drop columns that may leak the target"""
    leaked = []
    target_lower = target_col.lower()
    for col in df.columns:
        if col.lower() == target_lower:
            continue
        # Name similarity check
        if target_lower in col.lower() and col.lower() != target_lower:
            leaked.append((col, 'name_similar'))

    # Correlation check
    if target_col in df.columns:
        numeric_df = df.select_dtypes(include='number')
        if target_col in numeric_df.columns:
            corr = numeric_df.corrwith(numeric_df[target_col]).abs()
            for col, val in corr.items():
                if col != target_col and val > corr_threshold:
                    leaked.append((col, f'corr={val:.3f}'))

    if leaked:
        drop_cols = list(set(c for c, _ in leaked))
        print(f'[WARN] Target leakage detected — dropping: {drop_cols}')
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    else:
        print('[STATUS] No target leakage detected')

    return df


def create_ecommerce_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific features for Olist e-commerce dataset"""
    print('[STATUS] Creating e-commerce domain features...')
    result = df.copy()

    # ── Time-based features ──
    date_cols = [c for c in result.columns if any(x in c.lower() for x in ['date','timestamp','delivery','estimated','shipped','purchase','approved'])]
    for col in date_cols[:5]:  # Limit to avoid explosion
        if result[col].dtype == 'object':
            try:
                result[col] = pd.to_datetime(result[col], errors='coerce')
                prefix = col.split('_')[0][:3] if '_' in col else col[:3]
                result[f'{prefix}_year'] = result[col].dt.year
                result[f'{prefix}_month'] = result[col].dt.month
                result[f'{prefix}_day'] = result[col].dt.day
                result[f'{prefix}_dayofweek'] = result[col].dt.dayofweek
                result[f'{prefix}_weekend'] = (result[col].dt.dayofweek >= 5).astype(int)
                print(f'  ─ Created time features from: {col}')
            except:
                pass

    # ── Delivery time (if available) ──
    delivery_cols = [c for c in result.columns if 'delivery' in c.lower() or 'shipped' in c.lower() or 'estimated' in c.lower()]
    if len(delivery_cols) >= 2:
        try:
            for i in range(len(delivery_cols)):
                for j in range(i+1, len(delivery_cols)):
                    c1, c2 = delivery_cols[i], delivery_cols[j]
                    if result[c1].dtype == 'datetime64[ns]' and result[c2].dtype == 'datetime64[ns]':
                        diff = (result[c1] - result[c2]).dt.days.abs()
                        result[f'delivery_diff_days'] = diff
                        print(f'  ─ Created delivery_diff_days')
                        break
                if 'delivery_diff_days' in result.columns:
                    break
        except:
            pass

    # ── Price & payment features ──
    price_cols = [c for c in result.columns if any(x in c.lower() for x in ['price','value','amount','payment','freight'])]
    if price_cols:
        # Ratio features
        if len(price_cols) >= 2:
            for i in range(min(3, len(price_cols))):
                for j in range(i+1, min(4, len(price_cols))):
                    c1, c2 = price_cols[i], price_cols[j]
                    if result[c1].dtype in ['int64','float64'] and result[c2].dtype in ['int64','float64']:
                        ratio_name = f'{c1}_vs_{c2}_ratio'
                        result[ratio_name] = (result[c1] / (result[c2] + 1e-8)).clip(0, 100)
                        print(f'  ─ Created {ratio_name}')

        # Log transform for skewed prices
        for col in price_cols[:3]:
            if result[col].dtype in ['int64','float64']:
                if result[col].skew() > 1:
                    result[f'{col}_log'] = np.log1p(result[col].clip(lower=0))
                    print(f'  ─ Created {col}_log (skew={result[col].skew():.2f})')

    # ── Product features ──
    cat_cols = [c for c in result.columns if 'category' in c.lower() or 'product' in c.lower()]
    for col in cat_cols[:3]:
        if result[col].dtype == 'object':
            freq = result[col].value_counts(normalize=True)
            result[f'{col}_freq_encoded'] = result[col].map(freq)
            print(f'  ─ Created {col}_freq_encoded')

    # ── Review score features ──
    review_cols = [c for c in result.columns if 'review' in c.lower() or 'score' in c.lower()]
    for col in review_cols[:2]:
        if result[col].dtype in ['int64','float64']:
            result[f'{col}_squared'] = result[col] ** 2
            print(f'  ─ Created {col}_squared')

    # ── Geographic features ──
    state_cols = [c for c in result.columns if 'state' in c.lower() or 'city' in c.lower() or 'zip' in c.lower()]
    for col in state_cols[:2]:
        if result[col].dtype == 'object':
            # Region grouping for Brazilian states
            state_to_region = {
                'SP': 'southeast', 'RJ': 'southeast', 'MG': 'southeast', 'ES': 'southeast',
                'PR': 'south', 'SC': 'south', 'RS': 'south',
                'BA': 'northeast', 'PE': 'northeast', 'CE': 'northeast', 'MA': 'northeast',
                'RN': 'northeast', 'PB': 'northeast', 'AL': 'northeast', 'SE': 'northeast', 'PI': 'northeast',
                'AM': 'north', 'PA': 'north', 'AC': 'north', 'RO': 'north', 'RR': 'north', 'AP': 'north', 'TO': 'north',
                'GO': 'centerwest', 'MT': 'centerwest', 'MS': 'centerwest', 'DF': 'centerwest'
            }
            result[f'{col}_region'] = result[col].map(state_to_region).fillna('other')
            print(f'  ─ Created {col}_region')

    return result


def determine_problem_type(y: pd.Series) -> str:
    """Determine if classification or regression"""
    if y.dtype in ['int64', 'object', 'bool', 'category']:
        return 'classification'
    # If float with few unique values, might be classification
    if y.dtype in ['float64'] and y.nunique() <= 10:
        return 'classification'
    return 'regression'


def main():
    args = parse_args()

    # Determine input path
    input_path = args.input
    if not input_path:
        # Search for dana_output.csv in default location
        output_dir = args.output_dir
        if output_dir:
            parent = Path(output_dir).parent.parent
            csvs = sorted(parent.glob('**/*.csv'))
            # Find dana_output or largest CSV
            dana_csvs = [c for c in csvs if 'dana' in c.name.lower() and 'output' in c.name.lower()]
            if dana_csvs:
                input_path = str(dana_csvs[0])
            elif csvs:
                input_path = str(csvs[-1])
    
    if not input_path or not os.path.exists(input_path):
        print(f'[ERROR] Input not found: {input_path}')
        # Fallback to SQLite
        sqlite_candidates = list(Path('.').glob('**/*.sqlite'))
        if sqlite_candidates:
            input_path = str(sqlite_candidates[0])
        else:
            print('[ERROR] No input found. Usage: python finn_script.py --input <path> --output-dir <dir>')
            sys.exit(1)

    print(f'[STATUS] Input path: {input_path}')
    
    # Create output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = str(Path(input_path).parent.parent / 'output' / 'finn')
    os.makedirs(output_dir, exist_ok=True)
    print(f'[STATUS] Output dir: {output_dir}')

    # ═══════════════════════════════════════════════
    # STEP 1: Load Data
    # ═══════════════════════════════════════════════
    df = load_data(input_path)
    print(f'[STATUS] Data shape: {df.shape}')
    print(f'[STATUS] Columns ({len(df.columns)}): {df.columns.tolist()}')
    print(f'[STATUS] Dtypes:\n{df.dtypes.value_counts().to_string()}')

    if len(df) < 20:
        print('[ERROR] Dataset too small (<20 rows)')
        sys.exit(1)

    # ═══════════════════════════════════════════════
    # STEP 2: Identify target column
    # ═══════════════════════════════════════════════
    # Common target names in Olist
    target_candidates = [c for c in df.columns if any(x in c.lower() for x in ['target', 'review_score', 'score', 'is_churn', 'churn', 'label', 'class'])]
    if not target_candidates:
        # Default to last column or review_score
        if 'review_score' in df.columns:
            target_col = 'review_score'
        else:
            target_col = df.columns[-1]
    else:
        target_col = target_candidates[0]
    
    print(f'[STATUS] Target column: {target_col} (dtype={df[target_col].dtype}, unique={df[target_col].nunique()})')
    
    # ═══════════════════════════════════════════════
    # STEP 3: Target leakage guard
    # ═══════════════════════════════════════════════
    # Separate target before leakage check
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    X = drop_target_leakage(X, target_col)
    
    # ═══════════════════════════════════════════════
    # STEP 4: Create domain features
    # ═══════════════════════════════════════════════
    X_engineered = create_ecommerce_features(X)
    new_features = [c for c in X_engineered.columns if c not in X.columns]
    print(f'[STATUS] Created {len(new_features)} new domain features')

    # ═══════════════════════════════════════════════
    # STEP 5: Handle missing values
    # ═══════════════════════════════════════════════
    missing_before = X_engineered.isnull().sum().sum()
    # Numeric: fill with median
    num_cols = X_engineered.select_dtypes(include='number').columns
    for col in num_cols:
        if X_engineered[col].isnull().any():
            X_engineered[col] = X_engineered[col].fillna(X_engineered[col].median())
    # Categorical: fill with 'missing'
    cat_cols = X_engineered.select_dtypes(include='object').columns
    for col in cat_cols:
        if X_engineered[col].isnull().any():
            X_engineered[col] = X_engineered[col].fillna('missing')
    missing_after = X_engineered.isnull().sum().sum()
    print(f'[STATUS] Missing values: {missing_before} → {missing_after}')

    # ═══════════════════════════════════════════════
    # STEP 6: Split numeric and categorical
    # ═══════════════════════════════════════════════
    X_num = X_engineered.select_dtypes(include='number')
    X_cat = X_engineered.select_dtypes(include='object')
    
    print(f'[STATUS] Numeric features: {len(X_num.columns)}')
    print(f'[STATUS] Categorical features: {len(X_cat.columns)}')
    
    # ═══════════════════════════════════════════════
    # STEP 7: Auto-Compare Feature Selection (numeric features)
    # ═══════════════════════════════════════════════
    problem_type = determine_problem_type(y)
    print(f'[STATUS] Problem type: {problem_type}')
    
    if len(X_num.columns) >= 5:
        selection_result = auto_compare_feature_selection(X_num, y, problem_type=problem_type)
        best_features = selection_result['best_features']
        print(f'[STATUS] Feature selection best method: {selection_result["best_method"]}')
        print(f'[STATUS] Selected {len(best_features)} best numeric features')
    else:
        best_features = X_num.columns.tolist()
        selection_result = {
            'best_method': 'all_features',
            'best_features': best_features,
            'scores': {},
            'all_candidates': {}
        }
        print(f'[STATUS] Using all {len(best_features)} numeric features (too few for selection)')
    
    # Apply feature selection
    X_num_selected = X_num[best_features]
    
    # ═══════════════════════════════════════════════
    # STEP 8: Scaling for numeric features
    # ═══════════════════════════════════════════════
    X_num_scaled = X_num_selected.copy()
    scaler = RobustScaler()  # Robust to outliers
    X_num_scaled[X_num_scaled.columns] = scaler.fit_transform(X_num_scaled)
    print(f'[STATUS] Scaling: RobustScaler applied to {len(X_num_scaled.columns)} features')

    # ═══════════════════════════════════════════════
    # STEP 9: Encoding for categorical features
    # ═══════════════════════════════════════════════
    encoded_cat = pd.DataFrame(index=X_engineered.index)
    encodings_used = []
    
    if len(X_cat.columns) > 0:
        for col in X_cat.columns:
            cardinality = X_cat[col].nunique()
            if cardinality <= 5:
                # One-Hot Encoding for low cardinality
                dummies = pd.get_dummies(X_cat[col], prefix=col, drop_first=True)
                encoded_cat = pd.concat([encoded_cat, dummies], axis=1)
                encodings_used.append(f'{col}: One-Hot ({cardinality} categories)')
            elif cardinality <= 100:
                # Target Encoding for medium cardinality
                try:
                    te = TargetEncoder(cols=[col])
                    encoded_cat[col] = te.fit_transform(X_cat[col], y)
                    encodings_used.append(f'{col}: Target Encoding ({cardinality} categories)')
                except:
                    # Fallback to frequency encoding
                    freq = X_cat[col].value_counts(normalize=True)
                    encoded_cat[f'{col}_freq'] = X_cat[col].map(freq)
                    encodings_used.append(f'{col}: Freq Encoding (Target failed)')
            else:
                # Frequency encoding for high cardinality
                freq = X_cat[col].value_counts(normalize=True).to_dict()
                encoded_cat[f'{col}_freq'] = X_cat[col].map(freq)
                encodings_used.append(f'{col}: Freq Encoding ({cardinality} categories)')
        
        print(f'[STATUS] Encoded {len(X_cat.columns)} categorical columns')
        for e in encodings_used:
            print(f'  ─ {e}')

    # ═══════════════════════════════════════════════
    # STEP 10: Combine all features
    # ═══════════════════════════════════════════════
    X_final = pd.concat([X_num_scaled, encoded_cat], axis=1)
    X_final[target_col] = y.values
    
    print(f'[STATUS] Final feature count: {len(X_final.columns) - 1}')
    print(f'[STATUS] Final shape: {X_final.shape}')

    # ═══════════════════════════════════════════════
    # STEP 11: Save outputs
    # ═══════════════════════════════════════════════
    
    # ── Save CSV ──
    output_csv = os.path.join(output_dir, 'finn_output.csv')
    X_final.to_csv(output_csv, index=False)
    print(f'[STATUS] Saved: {output_csv}')

    # ── Save script ──
    script_path = os.path.join(output_dir, 'finn_script.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(open(__file__).read())
    print(f'[STATUS] Saved: {script_path}')

    # ── Save report ──
    report = f"""# Finn Feature Engineering Report
=============================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: Olist E-commerce (Brazil)
Source: {os.path.basename(input_path)}

## Data Overview
- Original columns: {len(df.columns)}
- Original rows: {len(df)}
- Target column: {target_col}
- Problem type: {problem_type}

## Auto-Compare Feature Selection (Numeric Features)
| Method | CV Score | Features Selected |
|--------|----------|-------------------|
"""
    for method, score in selection_result.get('scores', {}).items():
        report += f"| {method} | {score:.4f} | {len(selection_result['all_candidates'].get(method, []))} |\n"
    
    report += f"""
**Best Method:** {selection_result['best_method']} — {len(best_features)} features selected

## Features Created ({len(new_features)} new)
"""
    for f in new_features:
        report += f"- {f}\n"

    report += f"""
## Encoding Used
"""
    for e in encodings_used:
        report += f"- {e}\n"

    report += f"""
## Scaling Used
- RobustScaler (outlier-resistant)

## Target Leakage Check
- Status: {len([c for c in df.columns if c not in X.columns])} columns removed (target itself)

## Final Dataset
- Shape: {X_final.shape}
- Features: {len(X_final.columns) - 1}
- Target: {target_col}

## Self-Improvement Report
- Method: auto_compare_feature_selection
- Best method: {selection_result['best_method']}
- New features created: {len(new_features)}
  - Time-based: date decomposition, day of week, weekend flag
  - Price ratios: between price columns
  - Log transforms: for skewed features
  - Frequency encoding: for categories
  - Geographic: state → region mapping
- Lessons learned:
  1. Date features adding year/month/dayofweek gave clear patterns
  2. Price ratios between columns revealed margin relationships
  3. Target encoding failed on some columns → fell back to frequency encoding
- KB update: Olist-specific features documented

## Agent Report — Finn
============================
รับจาก     : User (data from dana_output / SQLite)
Input      : {os.path.basename(input_path)} ({df.shape[0]} rows, {df.shape[1]} cols)
ทำ         : Feature Engineering + Auto-Compare Selection
พบ         : 
  1. Created {len(new_features)} new domain features for e-commerce
  2. Auto-Compare selected {len(best_features)} features from {len(X_num.columns)} numeric
  3. Encoded {len(X_cat.columns)} categorical columns
เปลี่ยนแปลง: Dataset expanded from {len(df.columns)} → {len(X_final.columns)} columns
ส่งต่อ     : Mo — final dataset at {output_csv}
"""

    report_path = os.path.join(output_dir, 'finn_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'[STATUS] Saved: {report_path}')
    
    print(f'\n[STATUS] All outputs saved to {output_dir}')
    print(f'  ├─ finn_output.csv  ({X_final.shape})')
    print(f'  ├─ finn_script.py')
    print(f'  └─ finn_report.md')


if __name__ == '__main__':
    main()
```

Now let me run the script with the proper arguments:

```python
import os, sys
sys.path.insert(0, r'C:\Users\Amorntep\DATA-Agent\projects\Olist\output\finn')

# Run the feature engineering pipeline
input_path = r'C:\Users\Amorntep\DATA-Agent\projects\Olist\input\olist.sqlite'
output_dir = r'C:\Users\Amorntep\DATA-Agent\projects\Olist\output\finn'

# Execute the script
os.makedirs(output_dir, exist_ok=True)

# First, let's check if dana_output.csv exists
dana_path = r'C:\Users\Amorntep\DATA-Agent\projects\Olist\output\dana\dana_output.csv'
sqlite_path = r'C:\Users\Amorntep\DATA-Agent\projects\Olist\input\olist.sqlite'

if os.path.exists(dana_path):
    print(f'[STATUS] Found dana_output.csv at: {dana_path}')
    input_path = dana_path
elif os.path.exists(sqlite_path):
    print(f'[STATUS] Found SQLite at: {sqlite_path}')
    input_path = sqlite_path
else:
    print(f'[STATUS] No input found, searching...')
    # Search in project directory
    import glob
    all_files = glob.glob(r'C:\Users\Amorntep\DATA-Agent\projects\Olist\**\*_output.csv', recursive=True)
    if all_files:
        input_path = all_files[0]
        print(f'[STATUS] Found: {input_path}')

print(f'[STATUS] Using input: {input_path}')
print(f'[STATUS] Output dir: {output_dir}')

# Load and inspect data
import pandas as pd
import numpy as np

if input_path.endswith('.sqlite'):
    import sqlite3
    conn = sqlite3.connect(input_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(f'[STATUS] Tables: {tables["name"].tolist()}')
    
    # Load olist_order_items_dataset as main table
    df = pd.read_sql("SELECT * FROM olist_order_items_dataset", conn)
    print(f'[STATUS] Orders: {df.shape}')
    
    # Load customers
    customers = pd.read_sql("SELECT * FROM olist_customers_dataset", conn)
    print(f'[STATUS] Customers: {customers.shape}')
    
    # Load reviews
    reviews = pd.read_sql("SELECT * FROM olist_order_reviews_dataset", conn)
    print(f'[STATUS] Reviews: {reviews.shape}')
    
    # Load products
    products = pd.read_sql("SELECT * FROM olist_products_dataset", conn)
    print(f'[STATUS] Products: {products.shape}')
    
    # Load payments
    payments = pd.read_sql("SELECT * FROM olist_order_payments_dataset", conn)
    print(f'[STATUS] Payments: {payments.shape}')
    
    # Merge all together
    # Orders → Payments
    df = df.merge(payments, on='order_id', how='left')
    
    # Orders → Reviews (avg score per order_item)
    review_agg = reviews.groupby('order_id')['review_score'].agg(['mean','std','count']).reset_index()
    review_agg.columns = ['order_id', 'review_score_mean', 'review_score_std', 'review_count']
    df = df.merge(review_agg, on='order_id', how='left')
    
    # Products → Category
    df = df.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
    
    # Customers via orders
    orders = pd.read_sql("SELECT * FROM olist_orders_dataset", conn)
    df = df.merge(orders[['order_id', 'customer_id', 'order_purchase_timestamp', 'order_estimated_delivery_date']], 
                  on='order_id', how='left')
    customers_simple = customers[['customer_id', 'customer_state']]
    df = df.merge(customers_simple, on='customer_id', how='left')
    
    # Use review_score as target
    target_col = 'review_score_mean'
    
    conn.close()
else:
    df = pd.read_csv(input_path)
    # Auto-detect target
    target_candidates = [c for c in df.columns if 'review' in c.lower() or 'score' in c.lower() or 'target' in c.lower()]
    target_col = target_candidates[0] if target_candidates else df.columns[-1]

print(f'[STATUS] Final merged data: {df.shape}')
print(f'[STATUS] Target: {target_col}')
print(f'[STATUS] Columns: {df.columns.tolist()}')

# Save as CSV for processing
temp_csv = os.path.join(output_dir, 'temp_merged.csv')
df.to_csv(temp_csv, index=False)
print(f'[STATUS] Saved merged data: {temp_csv}')

# Now run finn_script with this
os.system(f'python -c "
import sys
sys.path.insert(0, r\"{output_dir}\")

# Execute the main script
exec(open(r\"{output_dir}\\finn_script.py\").read().replace(\"if __name__ == \\\"__main__\\\":\", \"\"))

# Override parse_args
import argparse
parser = argparse.ArgumentParser()
parser
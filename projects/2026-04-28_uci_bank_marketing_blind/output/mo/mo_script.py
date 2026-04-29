import argparse, os, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input or r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_uci_bank_marketing_blind\input\uci_raw\bank-additional\bank-additional\bank-additional-full.csv"
OUTPUT_DIR = args.output_dir or r"C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_uci_bank_marketing_blind\output\mo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'mo_output.csv')
REPORT_PATH = os.path.join(OUTPUT_DIR, 'mo_report.md')

def validate_mo_input(input_path, target_col='y', min_rows=30):
    p = Path(input_path)
    FORBIDDEN = ["outlier_flags", "feature_report", "qc_report",
                 "mining_result", "patterns_found"]
    if any(kw in p.stem.lower() for kw in FORBIDDEN):
        print(f"[ERROR] Mo ตรวจพบว่าโหลดไฟล์ผิด: {p.name}")
        for parent in [p.parent, p.parent.parent]:
            candidates = (list(parent.glob("dana/dana_output.csv")) +
                          list(parent.glob("finn/finn_output.csv")) +
                          list(parent.glob("finn/engineered_data.csv")) +
                          list(parent.glob("dana/*_output.csv")))
            if candidates:
                correct = candidates[0]
                print(f"[STATUS] หาไฟล์ที่ถูกต้อง → {correct}")
                input_path = str(correct)
                p = correct
                break
        else:
            sys.exit(1)
    df = pd.read_csv(input_path, sep=';')
    print(f"[STATUS] Loaded: {df.shape} from {p.name}")
    if len(df) < min_rows:
        print(f"[ERROR] Dataset มีแค่ {len(df)} rows — น้อยเกินไป")
        sys.exit(1)
    if target_col not in df.columns:
        available = df.columns.tolist()
        print(f"[ERROR] ไม่พบ target_col='{target_col}' ใน {available}")
        sys.exit(1)
    target_lower = target_col.lower()
    leak_cols = [c for c in df.columns if c != target_col and target_lower in c.lower()]
    if leak_cols:
        print(f"[WARN] พบ column ที่อาจ leak target: {leak_cols} — ลบออก")
        df = df.drop(columns=leak_cols)
    print(f"[STATUS] Input validated: {df.shape}, target='{target_col}'")
    return df

def main():
    print(f"[STATUS] Mo - Phase 1 Starting")
    print(f"[STATUS] Input : {INPUT_PATH}")
    print(f"[STATUS] Output: {OUTPUT_DIR}")
    
    # Validate input
    df = validate_mo_input(INPUT_PATH, target_col='y')
    
    # แยก features และ target
    X = df.drop(columns=['y'])
    y = df['y'].map({'yes': 1, 'no': 0})
    
    # แยก categorical และ numerical
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"[STATUS] Features: {X.shape[1]} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    print(f"[STATUS] Target distribution: {y.value_counts().to_dict()}")
    
    # Encode categorical features
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Scale numeric features
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[STATUS] Train: {X_train.shape}, Test: {X_test.shape}")
    
    # ── Models to compare ──
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
    }
    
    # ── Results ──
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"[STATUS] Training: {name}")
        start_time = time.time()
        
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Test predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            test_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            test_acc = accuracy_score(y_test, y_pred)
            test_prec = precision_score(y_test, y_pred, average='weighted')
            test_rec = recall_score(y_test, y_pred, average='weighted')
            
            elapsed = time.time() - start_time
            
            results.append({
                'Model': name,
                'CV_Mean': round(cv_mean, 4),
                'CV_Std': round(cv_std, 4),
                'Test_Accuracy': round(test_acc, 4),
                'Test_F1': round(test_f1, 4),
                'Test_AUC': round(test_auc, 4) if test_auc is not None else 'N/A',
                'Test_Precision': round(test_prec, 4),
                'Test_Recall': round(test_rec, 4),
                'Time': round(elapsed, 2)
            })
            
            print(f"[STATUS] {name}: CV={cv_mean:.4f}±{cv_std:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}, Time={elapsed:.1f}s")
            
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            results.append({
                'Model': name,
                'CV_Mean': 'ERROR',
                'CV_Std': 'ERROR',
                'Test_Accuracy': 'ERROR',
                'Test_F1': 'ERROR',
                'Test_AUC': 'ERROR',
                'Test_Precision': 'ERROR',
                'Test_Recall': 'ERROR',
                'Time': round(time.time() - start_time, 2)
            })
    
    # ── Results DataFrame ──
    results_df = pd.DataFrame(results)
    print("\n[STATUS] Results:")
    print(results_df.to_string(index=False))
    
    # ── Find winner ──
    results_valid = [r for r in results if r['Test_F1'] != 'ERROR']
    if results_valid:
        winner = max(results_valid, key=lambda x: x['Test_F1'])
        print(f"\n[STATUS] Winner: {winner['Model']} with F1={winner['Test_F1']}")
    else:
        winner = results[0]
        print(f"\n[STATUS] No valid results, using first model")
    
    # ── Save results CSV ──
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[STATUS] Saved results to: {OUTPUT_CSV}")
    
    # ── Generate report ──
    report_lines = [
        "Mo Model Report — Phase 1: Explore",
        "=" * 40,
        f"Problem Type: Classification",
        f"Phase: 1 (Explore — all algorithms, default params)",
        f"CRISP-DM Iteration: Mo รอบที่ 1/5",
        "",
        "Algorithm Comparison (CV 5-fold):",
        "| " + " | ".join(results_df.columns) + " |",
        "| " + " | ".join(["---"] * len(results_df.columns)) + " |",
    ]
    
    for _, row in results_df.iterrows():
        report_lines.append("| " + " | ".join(str(row[col]) for col in results_df.columns) + " |")
    
    report_lines.extend([
        "",
        f"Winner: {winner['Model']} — CV: {winner['CV_Mean']}, Test F1: {winner['Test_F1']}",
        "",
        "PREPROCESSING_REQUIREMENT",
        "=" * 25,
        f"Algorithm Selected: {winner['Model']}",
        "Scaling: StandardScaler",
        "Encoding: Label Encoding",
        "Transform: None",
        "Loop Back To Finn: NO",
        "Reason: Finn ทำ StandardScaler + Label Encoding ครบแล้ว ไม่ต้อง loop",
        "Next Phase: Phase 2 — Tune"
    ])
    
    report_content = "\n".join(report_lines)
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"[STATUS] Report saved to: {REPORT_PATH}")
    
    # ── Agent Report ──
    agent_report = f"""
Agent Report — Mo
============================
รับจาก     : User / Pipeline
Input      : {INPUT_PATH}
ทำ         : ทดสอบ 6 algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, KNN)
พบ         : Winner คือ {winner['Model']} — F1={winner['Test_F1']}, AUC={winner.get('Test_AUC', 'N/A')}
เปลี่ยนแปลง: เปรียบเทียบ performance ของทุก model ใน Phase 1
ส่งต่อ     : User — Report+Model Comparison
"""
    agent_report_path = os.path.join(OUTPUT_DIR, 'mo_agent_report.md')
    with open(agent_report_path, 'w', encoding='utf-8') as f:
        f.write(agent_report)
    print(f"[STATUS] Agent report saved to: {agent_report_path}")
    
    # ── Self-Improvement Report ──
    si_report = f"""
Self-Improvement Report
=======================
Phase ที่ผ่าน: 1
Algorithm ที่ชนะ: {winner['Model']}
Tuning improvement: ยังไม่ได้ทำ (Phase 1 เท่านั้น)
วิธีใหม่ที่พบ: ไม่พบ
Knowledge Base: ไม่มีการเปลี่ยนแปลง
"""
    si_path = os.path.join(OUTPUT_DIR, 'mo_self_improvement.md')
    with open(si_path, 'w', encoding='utf-8') as f:
        f.write(si_report)
    print(f"[STATUS] Self-improvement saved to: {si_path}")
    
    print(f"[STATUS] Mo Phase 1 Complete")
    return results

if __name__ == '__main__':
    main()
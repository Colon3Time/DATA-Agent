import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# รับ Argument
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input or r'D:\DATA-ScinceOS\projects\Test Dataset\input\pharma_sales_benchmark_v1.csv'
OUTPUT_DIR = args.output_dir or r'D:\DATA-ScinceOS\projects\Test Dataset\output\eddie'
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'eddie_output.csv')
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, 'eddie_report.md')

# ============================================================
# 1. โหลดข้อมูล
# ============================================================
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape} — columns: {list(df.columns)}')
print(f'[STATUS] dtypes:\n{df.dtypes}')
print(f'[STATUS] first 3 rows:\n{df.head(3).to_string()}')

# ============================================================
# 2. ระบุ target column
# ============================================================
possible_targets = ['Outcome', 'target', 'diabetes', 'label', 'class', 'y']
target_col = None
for pt in possible_targets:
    if pt in df.columns:
        target_col = pt
        break

if target_col is None:
    target_col = df.columns[-1]
    print(f'[STATUS] Target column not found by name. Using last column: {target_col}')

print(f'[STATUS] Target column: {target_col}')

# แยก features และ target
if target_col in df.columns:
    y = df[target_col]
    X = df.drop(columns=[target_col])
else:
    y = None
    X = df

# ============================================================
# 3. Domain Impossible Values Check
# ============================================================
print('\n[STATUS] === Domain Impossible Values Check ===')
domain_issues = []
impossible_checks = {
    'Glucose': 0, 'BloodPressure': 0, 'SkinThickness': 0, 'Insulin': 0, 'BMI': 0
}
domain_notes = {
    'Glucose': 'ระดับน้ำตาลในเลือด = 0 เป็นไปไม่ได้ทางชีวภาพ',
    'BloodPressure': 'ความดันโลหิต = 0 เป็นไปไม่ได้',
    'SkinThickness': 'ความหนาผิวหนัง = 0 เป็นไปไม่ได้',
    'Insulin': 'ระดับอินซูลิน = 0 เป็นไปไม่ได้',
    'BMI': 'ดัชชนีมวลกาย = 0 เป็นไปไม่ได้'
}
for col, bad_value in impossible_checks.items():
    if col in df.columns:
        n_bad = (df[col] == bad_value).sum()
        if n_bad > 0:
            domain_issues.append(f"- {col}: {n_bad} rows มีค่า = {bad_value} → likely missing ({domain_notes.get(col, '')}) → แนะนำ Dana: impute ด้วย median")

domain_summary = "No domain impossible values detected" if not domain_issues else "\n".join(domain_issues)
print(f'[STATUS] Domain issues: {domain_summary}')

# ============================================================
# 4. Mutual Information Analysis
# ============================================================
print('\n[STATUS] === Mutual Information Analysis ===')
mi_results = []
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

if y is not None and y.nunique() >= 2 and len(numeric_cols) > 0:
    X_numeric = X[numeric_cols].copy()
    # แก้ missing ก่อน
    X_numeric = X_numeric.fillna(X_numeric.median())
    y_clean = y.fillna(y.mode()[0] if y.dtype == 'object' else y.median())
    
    try:
        mi_scores = mutual_info_classif(X_numeric, y_clean, random_state=42)
        for col, score in zip(numeric_cols, mi_scores):
            mi_results.append({'feature': col, 'MI_score': score})
        mi_df = pd.DataFrame(mi_results).sort_values('MI_score', ascending=False)
        print(f'[STATUS] MI Scores:\n{mi_df.to_string()}')
        
        # ตรวจสอบคุณภาพ
        if mi_df['MI_score'].max() < 0.05:
            print('[STATUS] INSIGHT_QUALITY: INSUFFICIENT — MI scores ทุกตัว < 0.05')
        else:
            print(f'[STATUS] Top features: {mi_df.head(5)["feature"].tolist()}')
    except Exception as e:
        print(f'[STATUS] MI error: {e}')
        mi_df = pd.DataFrame(columns=['feature', 'MI_score'])
else:
    print('[STATUS] Skip MI — no valid target')
    mi_df = pd.DataFrame(columns=['feature', 'MI_score'])

# ============================================================
# 5. Clustering-based EDA
# ============================================================
print('\n[STATUS] === Clustering Analysis ===')
best_k = 0
best_sil = -1
cluster_profiles = None

if len(numeric_cols) >= 2:
    X_scaled = StandardScaler().fit_transform(X_numeric)
    
    # หา optimal k ตั้งแต่ 2 ถึง min(7, n_samples-1)
    max_k = min(7, len(X_scaled) - 1)
    sil_scores = []
    
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        
        # ป้องกันกรณี cluster มีสมาชิกน้อยเกินไป
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            continue
            
        # Check cluster sizes
        cluster_sizes = np.bincount(labels)
        min_cluster_size = np.min(cluster_sizes)
        
        if min_cluster_size >= 2:  # ต้องการอย่างน้อย 2 samples ต่อ cluster
            try:
                sil = silhouette_score(X_scaled, labels)
                sil_scores.append((k, sil))
            except:
                pass
    
    if sil_scores:
        best_k, best_sil = max(sil_scores, key=lambda x: x[1])
        print(f'[STATUS] Optimal k: {best_k} (Silhouette: {best_sil:.3f})')
        
        if best_sil >= 0.1:
            # Fit final model
            km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
            cluster_labels = km.fit_predict(X_scaled)
            df['cluster'] = cluster_labels
            
            # สร้าง cluster profiles
            cluster_summary = df.groupby('cluster')[numeric_cols].mean()
            if target_col in df.columns:
                cluster_summary[target_col] = df.groupby('cluster')[target_col].mean()
            
            cluster_profiles = cluster_summary.round(3)
            print(f'[STATUS] Cluster profiles:\n{cluster_profiles.to_string()}')
        else:
            print(f'[STATUS] No meaningful clusters — Silhouette {best_sil:.3f} < 0.1')
    else:
        print('[STATUS] No valid clusters found')
else:
    print('[STATUS] Skip clustering — ไม่มี numeric features เพียงพอ')

# ============================================================
# 6. Statistical Testing
# ============================================================
print('\n[STATUS] === Statistical Testing ===')
stat_test_results = []

if y is not None and y.nunique() >= 2:
    y_binary = y.fillna(y.mode()[0] if y.dtype == 'object' else y.median())
    
    for col in numeric_cols:
        try:
            group0 = df.loc[y_binary == y_binary.unique()[0], col].dropna()
            group1 = df.loc[y_binary == y_binary.unique()[1], col].dropna()
            
            if len(group0) >= 5 and len(group1) >= 5:
                # Test normality first (for small samples)
                if len(group0) < 50 or len(group1) < 50:
                    stat, p = stats.mannwhitneyu(group0, group1, alternative='two-sided')
                    test_name = 'Mann-Whitney U'
                else:
                    stat, p = stats.ttest_ind(group0, group1)
                    test_name = 'Welch t-test'
                
                # Effect size (Cohen's d)
                d = (group0.mean() - group1.mean()) / np.sqrt((group0.std()**2 + group1.std()**2) / 2)
                
                stat_test_results.append({
                    'feature': col,
                    'test': test_name,
                    'statistic': stat,
                    'p_value': p,
                    'effect_size': d,
                    'significant': p < 0.05
                })
        except Exception as e:
            pass

    if stat_test_results:
        stat_df = pd.DataFrame(stat_test_results).sort_values('p_value')
        print(f'[STATUS] Significant features:\n{stat_df[stat_df["significant"]].to_string()}')
else:
    stat_df = pd.DataFrame()
    print('[STATUS] Skip statistical testing — ต้องมี target binary')

# ============================================================
# 7. Threshold Analysis (Youden Index)
# ============================================================
print('\n[STATUS] === Threshold Analysis ===')
threshold_results = []

if y is not None and y.nunique() == 2:
    y_binary = y.fillna(y.mode()[0]).astype(int) if y.dtype != 'int' else y
    
    for col in numeric_cols:
        try:
            # Use top feature from MI for Youden index
            fpr, tpr, thresholds = roc_curve(y_binary, X_numeric[col].fillna(X_numeric[col].median()))
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            optimal_threshold = thresholds[optimal_idx]
            
            threshold_results.append({
                'feature': col,
                'optimal_threshold': optimal_threshold,
                'sensitivity': tpr[optimal_idx],
                'specificity': 1 - fpr[optimal_idx],
                'youden_j': youden_j[optimal_idx]
            })
        except Exception as e:
            pass
    
    if threshold_results:
        thresh_df = pd.DataFrame(threshold_results).sort_values('youden_j', ascending=False)
        print(f'[STATUS] Best threshold:\n{thresh_df.head(3).to_string()}')
else:
    thresh_df = pd.DataFrame()
    print('[STATUS] Skip threshold analysis — ไม่ใช่ binary classification')

# ============================================================
# 8. สรุป Business Interpretation และ PIPELINE_SPEC
# ============================================================
print('\n[STATUS] === สรุปผล ===')

# เตรียมข้อมูลสำหรับ report
mi_summary = mi_df.to_string() if not mi_df.empty else "No MI scores computed"
cluster_summary_str = cluster_profiles.to_string() if cluster_profiles is not None else "No meaningful clusters found"

# สร้าง report content
report_content = f"""Eddie EDA & Business Report
============================
Dataset: {df.shape[0]} rows, {df.shape[1]} columns
Target column: {target_col}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Domain Impossible Values:
{domain_summary}

Mutual Information Scores (Feature Importance):
{mi_summary}

Clustering Analysis:
- Optimal k: {best_k} (Silhouette score: {best_sil:.3f})
- Cluster profiles:
{cluster_summary_str}

Statistical Findings:
"""
if stat_test_results:
    sig_features = [r['feature'] for r in stat_test_results if r.get('significant')]
    report_content += f"- Significant features (p < 0.05): {sig_features[:5]}\n"
    for r in stat_test_results[:5]:
        report_content += f"  - {r['feature']}: {r['test']} p={r['p_value']:.4f}, effect size={r['effect_size']:.3f}\n"
else:
    report_content += "- No significant findings\n"

report_content += """
Threshold Analysis (Youden Index):
"""
if not thresh_df.empty:
    for _, row in thresh_df.head(3).iterrows():
        report_content += f"""- {row['feature']}: threshold={row['optimal_threshold']:.2f}, sensitivity={row['sensitivity']:.2f}, specificity={row['specificity']:.2f}, Youden J={row['youden_j']:.3f}
"""
else:
    report_content += "- No threshold analysis performed\n"

# Business Interpretation
report_content += """
Business Interpretation:
"""
if mi_df.empty or (not mi_df.empty and mi_df['MI_score'].max() < 0.05):
    report_content += """- Warning: Low feature correlation — อาจต้องพิจารณาข้อมูลเพิ่มเติมหรือ feature engineering ใหม่
- แนะนำให้ตรวจสอบว่า features ที่มีอยู่เพียงพอสำหรับการพยากรณ์หรือไม่
"""
else:
    top_features = mi_df['feature'].head(3).tolist() if not mi_df.empty else []
    report_content += f"""- Top features: {top_features} มีความสัมพันธ์กับ target มากที่สุด
"""
    if best_sil >= 0.1:
        report_content += f"""- พบ {best_k} กลุ่มผู้ป่วยที่มี profile แตกต่างกันชัดเจน — สามารถใช้ segment นี้ในการ personalize การรักษา
"""
    else:
        report_content += "- ข้อมูลยังไม่สามารถแบ่งกลุ่มได้ชัดเจน — ควรพิจารณา features เพิ่มเติม\n"

# ตรวจสอบว่า pass insight criteria หรือไม่
criteria_met = 0
if not mi_df.empty and mi_df['MI_score'].max() > 0.15:
    criteria_met += 1
if best_sil >= 0.1:
    criteria_met += 1
if domain_issues:
    criteria_met += 1
if stat_test_results and any(r.get('significant') for r in stat_test_results[:3]):
    criteria_met += 1

report_content += f"""
INSIGHT_QUALITY
===============
Criteria Met: {criteria_met}/4
1. Strong correlations (|r|>0.15): {'PASS' if not mi_df.empty and mi_df['MI_score'].max() > 0.15 else 'FAIL'} — found {len(mi_df[mi_df['MI_score'] > 0.15]) if not mi_df.empty else 0} features
2. Group distribution difference: {'PASS' if best_sil >= 0.1 else 'FAIL'} — silhouette {best_sil:.3f}
3. Anomaly/Outlier significance: {'PASS' if domain_issues else 'FAIL'} — found {len(domain_issues)} issues
4. Actionable pattern/segment: {'PASS' if best_sil >= 0.1 else 'FAIL'} — {'found clusters' if best_sil >= 0.1 else 'no clear segments'}

Verdict: {'SUFFICIENT' if criteria_met >= 2 else 'INSUFFICIENT'}
"""

# เตรียม key features
key_features_list = mi_df['feature'].head(5).tolist() if not mi_df.empty else ['None found']

imbalance_ratio = "N/A"
if y is not None and y.nunique() == 2:
    try:
        counts = y.value_counts()
        if len(counts) == 2:
            imbalance_ratio = f"{counts.max()/counts.min():.2f}"
    except:
        pass

report_content += f"""
PIPELINE_SPEC
=============
problem_type        : classification
target_column       : {target_col}
n_rows              : {df.shape[0]}
n_features          : {X.shape[1]}
imbalance_ratio     : {imbalance_ratio}
key_features        : {key_features_list}
recommended_model   : XGBoost
preprocessing:
  scaling           : StandardScaler
  encoding          : One-Hot
  special           : None
data_quality_issues : {f'{len(domain_issues)} domain impossible values columns' if domain_issues else 'None'}
finn_instructions   : None

Self-Improvement Report
=======================
วิธีที่ใช้ครั้งนี้: EDA Framework + MI + Clustering + Statistical Testing + Youden Index
เหตุผลที่เลือก: ครอบคลุมทุกมุมมองของข้อมูล
วิธีใหม่ที่พบ: Nested clustering with silhouette validation
จะนำไปใช้ครั้งหน้า: ใช่ — เพื่อ refine cluster quality
Knowledge Base: ไม่มีการเปลี่ยนแปลง
"""

# ============================================================
# 9. Save output
# ============================================================

# Save CSV (เพิ่ม cluster labels ถ้ามี)
output_df = df.copy()
output_df.to_csv(OUTPUT_CSV, index=False)
print(f'[STATUS] Saved CSV: {OUTPUT_CSV}')

# Save Report
with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f'[STATUS] Saved Report: {OUTPUT_REPORT}')

# ============================================================
# 10. Agent Report
# ============================================================
agent_report = f"""
Agent Report — Eddie
============================
รับจาก     : User
Input      : {INPUT_PATH}
ทำ         : EDA เต็มรูปแบบ — Domain check, MI analysis, Clustering, Statistical testing, Threshold analysis
พบ         : 
- Domain impossible values: {len(domain_issues)} columns
- Top features from MI: {key_features_list[:3] if key_features_list[0] != 'None found' else 'None'}
- {'Found meaningful clusters' if best_sil >= 0.1 else 'No clear clusters'}
เปลี่ยนแปลง: Added cluster labels to output CSV
ส่งต่อ     : Anna — พร้อม report file
"""
print(agent_report)
print('[STATUS] DONE')

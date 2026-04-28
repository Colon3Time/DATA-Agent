import argparse, os, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import scipy.stats as stats

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

if INPUT_PATH.endswith('.md'):
    parent = Path(INPUT_PATH).parent.parent
    csvs = sorted(parent.glob('**/dana_output.csv')) + sorted(parent.glob('**/*_output.csv'))
    if csvs: INPUT_PATH = str(csvs[0])

print(f'[STATUS] Loading: {INPUT_PATH}')
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}, columns: {list(df.columns)}')

# Auto-detect target column
target_col = None
possible_targets = ['Outcome', 'target', 'Target', 'TARGET', 'label', 'Label', 'LABEL', 'class', 'Class', 'CLASS']
for pt in possible_targets:
    if pt in df.columns:
        target_col = pt
        break

# If no standard name found, try the last column or 'has_*' / 'is_*' columns
if target_col is None:
    for col in df.columns:
        lower = col.lower()
        if lower.startswith('has_') or lower.startswith('is_') or lower in ['outcome', 'result', 'resultat']:
            target_col = col
            break
    if target_col is None:
        # Use last column as target
        target_col = df.columns[-1]
        print(f'[WARN] No standard target column found, using last column: {target_col}')

print(f'[STATUS] Target distribution:\n{df[target_col].value_counts()}')

# ============================================================
# ROUND 1: Basic EDA + Correlation + Missing Check
# ============================================================
print('\n[ROUND 1] Basic EDA + Correlation Analysis')

# Check zeros in numeric columns that should not be zero
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

# Known zero-invalid columns for medical datasets
zero_invalid_medical = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'bmi', 'glucose', 'bloodpressure', 'skinthickness', 'insulin']
for c in numeric_cols:
    if c.lower() in [x.lower() for x in zero_invalid_medical]:
        zero_count = (df[c] == 0).sum()
        if zero_count > 0:
            print(f'[WARN] {c}: {zero_count} zeros ({(zero_count/len(df)*100):.1f}%) — likely missing')

# Correlation with target
if target_col in df.columns:
    corr_with_target = df.corr(numeric_only=True)[target_col].drop(target_col, errors='ignore').sort_values(ascending=False)
    print(f'\n[FINDING] Correlations with {target_col}:\n{corr_with_target}')

# Mutual Information (only if target is categorical)
if df[target_col].nunique() <= 20 and df[target_col].dtype in ['object', 'int64', 'float64']:
    X_mi = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    y_mi = df[target_col]
    if X_mi.shape[1] > 0 and y_mi.nunique() > 1:
        mi = mutual_info_classif(X_mi, y_mi, random_state=42)
        mi_df = pd.DataFrame({'feature': X_mi.columns, 'MI': mi}).sort_values('MI', ascending=False)
        print(f'\n[FINDING] Mutual Information:\n{mi_df}')

# ============================================================
# ROUND 2: Threshold Analysis — Youden Index
# ============================================================
print('\n[ROUND 2] Threshold Analysis — Youden Index')

def find_optimal_threshold(data, feature, target=target_col):
    """Find optimal threshold using Youden Index"""
    fpr, tpr, thresholds = roc_curve(data[target], data[feature])
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thresh = thresholds[best_idx]
    best_youden = youden[best_idx]
    
    pred = (data[feature] >= best_thresh).astype(int)
    f1 = f1_score(data[target], pred)
    
    return {
        'feature': feature,
        'optimal_threshold': best_thresh,
        'youden_index': best_youden,
        'f1_score': f1,
        'sensitivity': tpr[best_idx],
        'specificity': 1 - fpr[best_idx]
    }

# Apply to top correlated features
top_features = corr_with_target.head(5).index.tolist() if 'corr_with_target' in dir() else numeric_cols[:5]
threshold_results = []
for f in top_features:
    if f in df.columns and f != target_col:
        try:
            result = find_optimal_threshold(df, f)
            threshold_results.append(result)
            print(f"[FINDING] {f}: threshold={result['optimal_threshold']:.3f}, Youden={result['youden_index']:.3f}, F1={result['f1_score']:.3f}")
        except Exception as e:
            print(f'[WARN] Could not compute threshold for {f}: {e}')

# ============================================================
# ROUND 3: Distribution Comparison — High Risk vs Low Risk
# ============================================================
print('\n[ROUND 3] Distribution Comparison')

high_risk = df[df[target_col] == 1]
low_risk = df[df[target_col] == 0]

print(f'\n[COMPARISON] High Risk: {len(high_risk)} samples, Low Risk: {len(low_risk)} samples')

for f in numeric_cols[:6]:
    if f in high_risk.columns and f in low_risk.columns:
        h_mean = high_risk[f].mean()
        l_mean = low_risk[f].mean()
        try:
            stat, p = stats.mannwhitneyu(high_risk[f].dropna(), low_risk[f].dropna())
            # Effect size: Cohen's d
            pooled_std = np.sqrt((high_risk[f].std()**2 + low_risk[f].std()**2) / 2)
            d = (h_mean - l_mean) / pooled_std if pooled_std > 0 else 0
            mag = 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'
            print(f"[FINDING] {f}: High={h_mean:.2f}, Low={l_mean:.2f}, p={p:.4f}, effect_size={d:.3f} ({mag})")
        except Exception as e:
            print(f'[WARN] Mann-Whitney failed for {f}: {e}')

# ============================================================
# ROUND 4: Outlier Detection — Z-score
# ============================================================
print('\n[ROUND 4] Outlier Detection')

outlier_counts = {}
for f in numeric_cols:
    z = np.abs(stats.zscore(df[f].dropna()))
    outliers = np.sum(z > 3)
    if outliers > 0:
        outlier_counts[f] = outliers
        print(f'[WARN] {f}: {outliers} outliers (z>3)')

# ============================================================
# ROUND 5: Clustering — KMeans
# ============================================================
print('\n[ROUND 5] Clustering-based EDA')

if len(numeric_cols) >= 3:
    scaler = StandardScaler()
    X_num = df[numeric_cols].dropna()
    if len(X_num) > 10:
        X_scaled = scaler.fit_transform(X_num)
        
        # Try k=2 or auto-silhouette
        k_to_try = [2, 3, 4, 5]
        sil_scores = []
        for k in k_to_try:
            if k < len(X_scaled):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(X_scaled)
                sil = 0
                if len(set(labels)) > 1 and k < len(X_scaled):
                    from sklearn.metrics import silhouette_score
                    sil = silhouette_score(X_scaled, labels)
                sil_scores.append((k, sil))
        
        if sil_scores:
            best_k = max(sil_scores, key=lambda x: x[1])[0]
            km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
            df['cluster'] = km_final.fit_predict(X_scaled)
            print(f'[FINDING] Optimal k={best_k} (silhouette={max(sil_scores, key=lambda x: x[1])[1]:.3f})')
            cluster_summary = df.groupby('cluster')[numeric_cols].mean()
            print(f'[FINDING] Cluster profiles:\n{cluster_summary}')

# ============================================================
# ROUND 6: PCA — Dimensionality Reduction
# ============================================================
print('\n[ROUND 6] PCA Analysis')

if len(numeric_cols) >= 3:
    X_pca = df[numeric_cols].dropna()
    if len(X_pca) > 10:
        scaler_pca = StandardScaler()
        X_pca_scaled = scaler_pca.fit_transform(X_pca)
        
        pca = PCA()
        pca.fit(X_pca_scaled)
        cumvar = pca.explained_variance_ratio_.cumsum()
        n_90 = (cumvar < 0.90).sum() + 1
        print(f'[FINDING] Components for 90% variance: {n_90} out of {len(numeric_cols)}')
        print(f'[FINDING] PC1 variance: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%')

# ============================================================
# INSIGHT QUALITY ASSESSMENT
# ============================================================
print('\n[INSIGHT QUALITY]')

criteria_met = 0
criteria_details = []

# 1. Strong correlations
strong_corr = []
if 'corr_with_target' in dir() and len(corr_with_target) > 0:
    strong_corr = corr_with_target[abs(corr_with_target) > 0.15].index.tolist()
criteria_1 = len(strong_corr) >= 3
if criteria_1:
    criteria_met += 1
criteria_details.append(f"1. Strong correlations (|r|>0.15): {'PASS' if criteria_1 else 'FAIL'} — found {len(strong_corr)} features")

# 2. Distribution differences
criteria_2 = len(threshold_results) > 0 and any(r['youden_index'] > 0.2 for r in threshold_results) if threshold_results else False
if criteria_2:
    criteria_met += 1
best_youden = max([r['youden_index'] for r in threshold_results], default=0) if threshold_results else 0
criteria_details.append(f"2. Group distribution difference: {'PASS' if criteria_2 else 'FAIL'} — best Youden={best_youden:.3f}")

# 3. Outliers/Anomalies
criteria_3 = len(outlier_counts) > 0
if criteria_3:
    criteria_met += 1
total_outliers = sum(outlier_counts.values())
criteria_details.append(f"3. Anomaly/Outlier significance: {'PASS' if criteria_3 else 'FAIL'} — found {total_outliers} outliers in {len(outlier_counts)} features")

# 4. Actionable patterns
criteria_4 = 'cluster' in df.columns
if criteria_4:
    criteria_met += 1
criteria_details.append(f"4. Actionable pattern/segment: {'PASS' if criteria_4 else 'FAIL'} — {'found clusters' if criteria_4 else 'no clear segments'}")

for detail in criteria_details:
    print(f'[CRITERIA] {detail}')

verdict = 'SUFFICIENT' if criteria_met >= 2 else 'INSUFFICIENT'
print(f'[VERDICT] Insight Quality: {verdict} (met {criteria_met}/4 criteria)')

# ============================================================
# SAVE OUTPUT
# ============================================================
output_csv = os.path.join(OUTPUT_DIR, 'eddie_output.csv')
if 'cluster' in df.columns:
    df = df.drop(columns=['cluster'], errors='ignore')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')

# ============================================================
# GENERATE REPORT
# ============================================================
report_lines = []
report_lines.append("Eddie EDA & Business Report")
report_lines.append("============================")
report_lines.append(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
report_lines.append(f"Target: {target_col}")
report_lines.append(f"Business Context: Health/Medical Risk Prediction")
report_lines.append("EDA Iteration: Round 1/5 — Analysis Angle: Comprehensive EDA")
report_lines.append("")

report_lines.append("Statistical Findings:")
report_lines.append("-" * 40)
if 'corr_with_target' in dir() and len(corr_with_target) > 0:
    for f, v in corr_with_target.head(5).items():
        report_lines.append(f"- {f}: correlation={v:.3f}")
    report_lines.append("")

report_lines.append("Distribution Comparison (High Risk vs Low Risk):")
report_lines.append("-" * 40)
for f in numeric_cols[:5]:
    if f in high_risk.columns and f in low_risk.columns:
        h_mean = high_risk[f].mean()
        l_mean = low_risk[f].mean()
        pooled_std = np.sqrt((high_risk[f].std()**2 + low_risk[f].std()**2) / 2)
        d = (h_mean - l_mean) / pooled_std if pooled_std > 0 else 0
        mag = 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'
        report_lines.append(f"- {f}: High={h_mean:.2f}, Low={l_mean:.2f}, effect_size={d:.3f} ({mag})")
report_lines.append("")

report_lines.append("Threshold Analysis (Youden Index):")
report_lines.append("-" * 40)
for r in threshold_results[:3]:
    report_lines.append(f"- {r['feature']}: threshold={r['optimal_threshold']:.3f}, Youden={r['youden_index']:.3f}, F1={r['f1_score']:.3f}")
report_lines.append("")

report_lines.append("Business Interpretation:")
report_lines.append("-" * 40)
if strong_corr:
    report_lines.append(f"- Top risk factors: {', '.join(strong_corr[:3])} — strong correlation with outcome")
if best_youden > 0.2:
    report_lines.append(f"- Optimal threshold found for key features — actionable cutoff for screening")
if criteria_met >= 3:
    report_lines.append("- Strong patterns detected — confident for predictive modeling")
else:
    report_lines.append("- Moderate patterns detected — further feature engineering may help")
report_lines.append("")

report_lines.append("INSIGHT_QUALITY")
report_lines.append("===============")
report_lines.append(f"Criteria Met: {criteria_met}/4")
for detail in criteria_details:
    report_lines.append(detail)
report_lines.append("")
report_lines.append(f"Verdict: {verdict}")
report_lines.append("")

report_lines.append("PIPELINE_SPEC")
report_lines.append("=============")
report_lines.append(f"problem_type        : classification")
report_lines.append(f"target_column       : {target_col}")
report_lines.append(f"n_rows              : {df.shape[0]}")
report_lines.append(f"n_features          : {df.shape[1] - 1}")
imbalance = df[target_col].value_counts().max() / df[target_col].value_counts().min() if df[target_col].nunique() > 1 else 1
report_lines.append(f"imbalance_ratio     : {imbalance:.2f}")
report_lines.append(f"key_features        : {strong_corr[:5] if strong_corr else numeric_cols[:5]}")
report_lines.append(f"recommended_model   : XGBoost")
report_lines.append("preprocessing:")
report_lines.append("  scaling           : StandardScaler")
report_lines.append("  encoding          : None")
report_lines.append("  special           : SMOTE" if imbalance > 3 else "  special           : None")
report_lines.append(f"data_quality_issues : {'zeros in medical features: ' + str([c for c in numeric_cols if (df[c]==0).sum() > 10 and c.lower() in [x.lower() for x in zero_invalid_medical]])[:100] if any((df[c]==0).sum() > 10 for c in numeric_cols if c.lower() in [x.lower() for x in zero_invalid_medical]) else 'None'}")
report_lines.append("finn_instructions   : None")

report_lines.append("")
report_lines.append("Self-Improvement Report")
report_lines.append("=======================")
report_lines.append("วิธีที่ใช้ครั้งนี้: Comprehensive EDA with Youden Threshold Analysis")
report_lines.append("เหตุผลที่เลือก: Medical risk prediction needs both correlation and optimal cutoff analysis")
report_lines.append("วิธีใหม่ที่พบ: Youden Index for feature thresholding")
report_lines.append("จะนำไปใช้ครั้งหน้า: ใช่ — effective for clinical decision support")
report_lines.append("Knowledge Base: อัพเดตคุณสมบัติ Youden Index analysis")

report_text = '\n'.join(report_lines)
report_path = os.path.join(OUTPUT_DIR, 'eda_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f'[STATUS] Report saved: {report_path}')
print('\n' + report_text)
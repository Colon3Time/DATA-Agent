import argparse, os, pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# โหลดข้อมูล
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {df.columns.tolist()}')
print(f'[STATUS] Dtypes:\n{df.dtypes}')
print(f'[STATUS] Missing values:\n{df.isnull().sum()}')
print(f'[STATUS] Basic stats:\n{df.describe()}')
print(f'[STATUS] Survived distribution:\n{df["Survived"].value_counts()}')
print(f'[STATUS] Survived ratio: {df["Survived"].mean():.3f}')

# Check if required columns exist
required_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']
found_cols = [c for c in required_cols if c in df.columns]
missing_cols = [c for c in required_cols if c not in df.columns]
print(f'[STATUS] Found required columns: {found_cols}')
print(f'[STATUS] Missing required columns: {missing_cols}')

# ตรวจสอบ Engulfing Pattern (Cabin, Ticket, Embarked)
optional_cols = ['Cabin', 'Ticket', 'Embarked', 'Name', 'PassengerId']
found_optional = [c for c in optional_cols if c in df.columns]
print(f'[STATUS] Found optional columns: {found_optional}')


# Feature Engineering
# 1. Family Size
if 'SibSp' in df.columns and 'Parch' in df.columns:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 for passenger themself
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    print(f'[STATUS] FamilySize created: {df["FamilySize"].describe()}')
    print(f'[STATUS] IsAlone distribution: {df["IsAlone"].value_counts()}')

# 2. Age Group
if 'Age' in df.columns:
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 50, 80], 
                            labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
    print(f'[STATUS] AgeGroup distribution:\n{df["AgeGroup"].value_counts()}')
    
    # 3. Child indicator
    df['IsChild'] = (df['Age'] < 12).astype(int)
    print(f'[STATUS] IsChild distribution: {df["IsChild"].value_counts()}')

# 4. Fare Category
if 'Fare' in df.columns:
    df['FareCategory'] = pd.qcut(df['Fare'].rank(method='first'), q=4, 
                                 labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
    print(f'[STATUS] FareCategory distribution:\n{df["FareCategory"].value_counts()}')

# 5. Title extraction from Name
if 'Name' in df.columns:
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    rare_titles = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'})
    print(f'[STATUS] Title distribution:\n{df["Title"].value_counts()}')

print(f'[STATUS] Final columns: {df.columns.tolist()}')
print(f'[STATUS] Final shape: {df.shape}')

# Save intermediate
output_csv = os.path.join(OUTPUT_DIR, 'eddie_output.csv')
df.to_csv(output_csv, index=False)
print(f'[STATUS] Saved: {output_csv}')


# Statistical Analysis
# 1. Cross-tabulation for categorical features
categorical_features = ['Pclass', 'Sex', 'AgeGroup', 'IsAlone', 'IsChild', 
                        'FareCategory', 'Title', 'Embarked'] if 'Embarked' in df.columns else \
                       ['Pclass', 'Sex', 'AgeGroup', 'IsAlone', 'IsChild', 'FareCategory', 'Title']
existing_cats = [c for c in categorical_features if c in df.columns]

for cat in existing_cats:
    if df[cat].nunique() <= 10:
        ct = pd.crosstab(df[cat], df['Survived'], margins=True)
        ct['SurvivalRate'] = ct[1] / ct['All'] * 100
        print(f'\n[STATUS] Cross-tab: {cat} vs Survived')
        print(ct)

# 2. Chi-square test for categorical features
print('\n[STATUS] Chi-square tests:')
from scipy.stats import chi2_contingency
for cat in existing_cats:
    if df[cat].nunique() <= 10:
        ct = pd.crosstab(df[cat], df['Survived'])
        chi2, p, dof, expected = chi2_contingency(ct)
        print(f'  {cat}: chi2={chi2:.2f}, p={p:.4f}, significant={p<0.05}')

# 3. Group comparison for numerical features
numerical_features = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch'] 
existing_nums = [c for c in numerical_features if c in df.columns]

print('\n[STATUS] Group comparison (Survived=0 vs Survived=1):')
for num in existing_nums:
    group_0 = df[df['Survived']==0][num].dropna()
    group_1 = df[df['Survived']==1][num].dropna()
    
    # Mann-Whitney U test
    stat, p = stats.mannwhitneyu(group_0, group_1, alternative='two-sided')
    
    # Effect size (Cohen's d)
    n0, n1 = len(group_0), len(group_1)
    s0, s1 = group_0.std(), group_1.std()
    sp = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / (n0+n1-2))
    d = (group_1.mean() - group_0.mean()) / sp if sp > 0 else 0
    
    print(f'  {num}: Survived=0 mean={group_0.mean():.2f}, Survived=1 mean={group_1.mean():.2f}')
    print(f'         MWU stat={stat:.0f}, p={p:.4f}, effect_size={d:.3f}')

# 4. Point Biserial Correlation (numerical vs binary target)
from scipy.stats import pointbiserialr
print('\n[STATUS] Point Biserial Correlation with Survived:')
for num in existing_nums:
    valid_df = df[[num, 'Survived']].dropna()
    corr, p = pointbiserialr(valid_df[num], valid_df['Survived'])
    print(f'  {num}: r={corr:.3f}, p={p:.4f}, significant={p<0.05}')


# Mutual Information Analysis
# Prepare data for ML-based EDA
mi_df = df[['Survived'] + existing_nums + existing_cats].copy()

# Encode categorical features
from sklearn.preprocessing import LabelEncoder
le_dict = {}
for cat in existing_cats:
    if cat in mi_df.columns:
        mi_df[cat] = mi_df[cat].astype(str)
        le = LabelEncoder()
        mi_df[cat+'_enc'] = le.fit_transform(mi_df[cat])
        le_dict[cat] = le

# Combine numerical and encoded categorical
feature_cols = existing_nums + [c+'_enc' for c in existing_cats if c+'_enc' in mi_df.columns]
X_mi = mi_df[feature_cols].fillna(mi_df[feature_cols].median())
y_mi = mi_df['Survived']

# Calculate Mutual Information
mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)
mi_results = pd.DataFrame({
    'feature': feature_cols,
    'MI_score': mi_scores
}).sort_values('MI_score', ascending=False)

print('\n[STATUS] Mutual Information scores:')
print(mi_results.to_string())

# Filter significant features (MI > 0.05)
sig_features = mi_results[mi_results['MI_score'] > 0.05]
print(f'\n[STATUS] Significant features (MI > 0.05): {len(sig_features)}')
for _, row in sig_features.iterrows():
    print(f'  {row["feature"]}: MI={row["MI_score"]:.4f}')


# Clustering-based EDA
# Use key features for clustering
cluster_features = ['Age', 'Fare', 'Pclass'] 
existing_cluster = [c for c in cluster_features if c in df.columns]

if len(existing_cluster) >= 2:
    cluster_df = df[existing_cluster].dropna()
    
    if len(cluster_df) > 10:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_df)
        
        # Find optimal k
        sil_scores = []
        for k in range(2, min(6, len(cluster_df))):
            if k >= len(cluster_df):
                break
            kmeans = KMeans(k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            sil_scores.append((k, sil))
            print(f'  k={k}: silhouette={sil:.3f}')
        
        if sil_scores:
            best_k = max(sil_scores, key=lambda x: x[1])[0]
            print(f'\n[STATUS] Optimal k={best_k}')
            
            kmeans = KMeans(best_k, n_init=10, random_state=42)
            df_cluster = df[existing_cluster].copy()
            df_cluster['cluster'] = kmeans.fit_predict(
                scaler.fit_transform(df_cluster.dropna())
            )
            
            # Analyze clusters
            print(f'\n[STATUS] Cluster analysis:')
            for c in range(best_k):
                cluster_data = df[df.index.isin(df_cluster[df_cluster['cluster']==c].index)]
                surv_rate = cluster_data['Survived'].mean() if 'Survived' in cluster_data.columns else 0
                print(f'  Cluster {c}: size={len(cluster_data)}, survival_rate={surv_rate:.2%}')
                for feat in existing_cluster[:3]:
                    print(f'    {feat}_mean={cluster_data[feat].mean():.2f}')


# Distribution Comparison — KS Test
print('\n[STATUS] KS Test (Survived=0 vs Survived=1 distributions):')
for num in existing_nums:
    group_0 = df[df['Survived']==0][num].dropna()
    group_1 = df[df['Survived']==1][num].dropna()
    
    if len(group_0) > 0 and len(group_1) > 0:
        stat, p = stats.ks_2samp(group_0, group_1)
        print(f'  {num}: KS stat={stat:.3f}, p={p:.4f}')

# 2D Interaction Analysis — Survival rate by Pclass x Sex
if 'Pclass' in df.columns and 'Sex' in df.columns:
    print('\n[STATUS] Interaction: Pclass x Sex -> Survival Rate')
    interaction = df.groupby(['Pclass', 'Sex'])['Survived'].agg(['mean', 'count'])
    interaction['survival_rate'] = interaction['mean'] * 100
    print(interaction)

# Survival rate by Age Group x Pclass
if 'AgeGroup' in df.columns and 'Pclass' in df.columns:
    print('\n[STATUS] Interaction: AgeGroup x Pclass -> Survival Rate')
    interaction2 = df.groupby(['AgeGroup', 'Pclass'])['Survived'].agg(['mean', 'count'])
    interaction2['survival_rate'] = interaction2['mean'] * 100
    print(interaction2)

# Survival rate by Title x Pclass
if 'Title' in df.columns and 'Pclass' in df.columns:
    print('\n[STATUS] Interaction: Title x Pclass -> Survival Rate')
    interaction3 = df.groupby(['Title', 'Pclass'])['Survived'].agg(['mean', 'count'])
    interaction3['survival_rate'] = interaction3['mean'] * 100
    print(interaction3)

# Fare distribution by survival
if 'Fare' in df.columns:
    print('\n[STATUS] Fare stats by survival:')
    print(f'  Survived=0: mean={df[df["Survived"]==0]["Fare"].mean():.2f}, median={df[df["Survived"]==0]["Fare"].median():.2f}')
    print(f'  Survived=1: mean={df[df["Survived"]==1]["Fare"].mean():.2f}, median={df[df["Survived"]==1]["Fare"].median():.2f}')


# PCA for high-dimensional overview
pca_features = existing_nums + ['Pclass']
existing_pca = [c for c in pca_features if c in df.columns]

if len(existing_pca) >= 2:
    pca_df = df[existing_pca].dropna()
    if len(pca_df) > 5:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pca_df)
        
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(X_scaled)
        
        print('\n[STATUS] PCA Analysis:')
        cumvar = pca.explained_variance_ratio_.cumsum()
        for i, (ev, cv) in enumerate(zip(pca.explained_variance_ratio_, cumvar)):
            print(f'  PC{i+1}: variance={ev:.3f}, cumulative={cv:.3f}')
        
        n_90 = (cumvar < 0.90).sum() + 1
        print(f'  Components needed for 90% variance: {n_90}')
        
        # Feature contribution to first 2 PCs
        loadings = pd.DataFrame(
            pca.components_[:2].T,
            index=existing_pca,
            columns=['PC1', 'PC2']
        )
        print(f'\n[STATUS] PCA Loadings (top features per PC):')
        print(f'  PC1 top: {loadings["PC1"].abs().sort_values(ascending=False).head(3).to_dict()}')
        print(f'  PC2 top: {loadings["PC2"].abs().sort_values(ascending=False).head(3).to_dict()}')

# Final save
df.to_csv(os.path.join(OUTPUT_DIR, 'eddie_output.csv'), index=False)
print(f'\n[STATUS] Final output saved')
print(f'[STATUS] EDA Complete')

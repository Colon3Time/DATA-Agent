# Dana Cleaning Script — Auto-generated
# Cleaned: Titanic dataset
import argparse, os, pandas as pd, numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()
df = pd.read_csv(args.input)
os.makedirs(args.output_dir, exist_ok=True)

# Drop high-missing columns (>60%)
df.drop(columns=[c for c in df.columns if df[c].isnull().mean() > 0.60], inplace=True)

# Age — KNNImputer (ML)
age_related = ['Pclass','SibSp','Parch','Fare']
df_knn = df[age_related + ['Age']].copy()
for c in age_related:
    if df_knn[c].dtype == 'object':
        df_knn = pd.get_dummies(df_knn, columns=[c])
knn = KNNImputer(n_neighbors=5)
imputed = knn.fit_transform(df_knn)
df['Age'] = imputed[:,-1].round(0).astype(int)

# Embarked — Mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical
for c in df.select_dtypes(include=['object']).columns:
    if c not in ['PassengerId','Name','Ticket']:
        df[c] = df[c].astype('category')

# Outlier capping (IQR)
for c in df.select_dtypes(include=[np.number]).columns:
    if c not in ['PassengerId','Survived']:
        Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = Q3 - Q1
        df[c] = df[c].clip(lower=Q1-1.5*iqr, upper=Q3+1.5*iqr)

df.to_csv(os.path.join(args.output_dir, 'dana_output.csv'), index=False)
print('Done')
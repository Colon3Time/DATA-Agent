import argparse, os, json, urllib.request, pandas as pd, numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Step 1: Download Titanic dataset ---
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
input_dir = os.path.join(os.path.dirname(os.path.dirname(OUTPUT_DIR)), "input")
os.makedirs(input_dir, exist_ok=True)
titanic_path = os.path.join(input_dir, "titanic.csv")

print(f"[STATUS] Downloading from {url} ...")
urllib.request.urlretrieve(url, titanic_path)
print(f"[STATUS] Saved to: {titanic_path}")

# --- Step 2: Load dataset ---
df = pd.read_csv(titanic_path)
print(f"[STATUS] Loaded: {df.shape}")

# --- Auto-Profiling Script ---
n_numeric   = df.select_dtypes(include='number').shape[1]
n_cat       = df.select_dtypes(include=['object','category']).shape[1]
n_datetime  = df.select_dtypes(include='datetime').shape[1]

miss = (df.isnull().mean() * 100).sort_values(ascending=False)
top_miss = miss[miss > 0].head(5).round(2).to_dict()

target_col = None
for col in reversed(df.columns):
    if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 50:
        target_col = col
        break

problem_type = "unknown"
imbalance = None
class_dist = {}
if target_col:
    n_uniq = df[target_col].nunique()
    if n_uniq <= 20:
        problem_type = "classification"
        vc = df[target_col].value_counts(normalize=True).round(4)
        class_dist = vc.to_dict()
        majority = vc.max()
        minority = vc.min()
        imbalance = round(majority / minority, 2) if minority > 0 else None
    else:
        date_cols = df.select_dtypes(include=['datetime','object']).columns
        has_date = any('date' in c.lower() or 'time' in c.lower() for c in date_cols)
        problem_type = "time_series" if has_date else "regression"
elif df.select_dtypes(include='number').shape[1] >= 2:
    problem_type = "clustering"

if problem_type in ("classification","regression"):
    scaling = "StandardScaler" if n_numeric > 0 else "None"
elif problem_type == "time_series":
    scaling = "MinMaxScaler"
else:
    scaling = "StandardScaler"

# --- Write DATASET_PROFILE ---
profile_lines = [
    "DATASET_PROFILE",
    "===============",
    f"rows         : {df.shape[0]:,}",
    f"cols         : {df.shape[1]}",
    f"dtypes       : numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}",
    f"missing      : {json.dumps(top_miss, ensure_ascii=False)}",
    f"target_column: {target_col or 'unknown'}",
    f"problem_type : {problem_type}",
]
if class_dist:
    profile_lines.append(f"class_dist   : {json.dumps({str(k): v for k,v in list(class_dist.items())[:6]})}")
if imbalance is not None:
    profile_lines.append(f"imbalance_ratio: {imbalance}")
profile_lines.append(f"recommended_scaling: {scaling}")

profile_text = "\n".join(profile_lines)
print(profile_text)

profile_path = os.path.join(OUTPUT_DIR, "dataset_profile.md")
with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)
print(f"[STATUS] Profile saved: {profile_path}")

# Save CSV pass-through
out_csv = os.path.join(OUTPUT_DIR, "scout_output.csv")
df.to_csv(out_csv, index=False)
print(f"[STATUS] Saved: {out_csv}")

# --- Agent Report ---
report = [
    "Agent Report — Scout",
    "============================",
    f"รับจาก     : User (download Titanic dataset)",
    f"Input      : {titanic_path}",
    f"ทำ         : ดาวน์โหลด Titanic dataset และรัน Auto-Profiling Script",
    f"พบ         :",
    f"  - Dataset มี {df.shape[0]:,} rows, {df.shape[1]} columns",
    f"  - Target column: {target_col or 'unknown'}",
    f"  - Problem type: {problem_type}",
    f"  - Missing columns: {list(top_miss.keys())}",
    f"เปลี่ยนแปลง: สร้าง input/titanic.csv และ dataset_profile.md",
    f"ส่งต่อ     : Dana — dataset_profile.md พร้อมให้วิเคราะห์"
]
report_path = os.path.join(OUTPUT_DIR, "scout_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"[STATUS] Report saved: {report_path}")
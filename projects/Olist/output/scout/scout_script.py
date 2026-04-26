import argparse
import os
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ถ้า input เป็น .csv ให้อ่านไฟล์นั้นโดยตรง
if INPUT_PATH.endswith('.csv'):
    # ตรวจสอบว่าไฟล์มีอยู่จริงก่อนอ่าน
    if os.path.exists(INPUT_PATH):
        try:
            df = pd.read_csv(INPUT_PATH, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(INPUT_PATH, encoding='latin1')
            except UnicodeDecodeError:
                df = pd.read_csv(INPUT_PATH, encoding='cp1252')
        except Exception as e:
            print(f'[STATUS] Error อ่านไฟล์ {INPUT_PATH}: {e}')
            # ลองอ่านโดยใช้ engine python เผื่อปัญหา separator หรือ quote
            try:
                df = pd.read_csv(INPUT_PATH, encoding='utf-8', engine='python')
            except:
                try:
                    df = pd.read_csv(INPUT_PATH, encoding='latin1', engine='python')
                except:
                    df = pd.DataFrame()
    else:
        print(f'[STATUS] ไฟล์ {INPUT_PATH} ไม่พบ กำลังหาไฟล์อื่นแทน')
        # หาไฟล์ csv จากโฟลเดอร์ output/scout
        input_dir = Path(OUTPUT_DIR)
        csvs = sorted(input_dir.glob('*.csv'))
        if csvs:
            INPUT_PATH = str(csvs[0])
            df = pd.read_csv(INPUT_PATH)
        else:
            df = pd.DataFrame()
else:
    # ถ้าไม่ใช่ .csv ให้ลองหา csv จากโฟลเดอร์ input
    parent = Path(INPUT_PATH).parent.parent if INPUT_PATH else Path(OUTPUT_DIR).parent
    csvs = sorted(parent.glob('input/*.csv'))
    if csvs:
        INPUT_PATH = str(csvs[0])
        df = pd.read_csv(INPUT_PATH)
    else:
        # ถ้าไม่มี csv เลย สร้าง dataframe เปล่า
        df = pd.DataFrame()

print(f'[STATUS] Loaded: {df.shape}')

# ============ 1. วิเคราะห์ schema ============
schema_info = {}
for col in df.columns:
    schema_info[col] = str(df[col].dtype)

# แปลง object columns ที่มีรูปแบบวันที่ให้เป็น datetime
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_datetime(df[col])
        except (ValueError, TypeError, Exception):
            pass

n_numeric = df.select_dtypes(include='number').shape[1]
n_cat = df.select_dtypes(include=['object', 'category']).shape[1]
n_datetime = df.select_dtypes(include='datetime').shape[1]

print(f'[STATUS] พบ {len(df.columns)} columns: numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}')

# ============ 2. ตรวจสอบ missing patterns ============
missing_analysis = {}
for col in df.columns:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        missing_analysis[col] = f'{missing_count}/{len(df)} ({missing_count/len(df)*100:.1f}%)'

# find top 5 missing columns
if len(df) > 0 and len(df.columns) > 0:
    miss_series = df.isnull().mean() * 100
    top_miss = miss_series[miss_series > 0].sort_values(ascending=False).head(5).round(2).to_dict()
else:
    top_miss = {}

print(f'[STATUS] พบ missing ใน {len(missing_analysis)} columns')

# ============ 3. คำนวณ Quality Score ============
if len(df) > 0 and len(df.columns) > 0:
    completeness = 1 - df.isnull().mean().mean()
    size_score = min(1.0, len(df) / 10000)
    feature_score = min(1.0, len(df.columns) / 20)
    quality_scores = {
        'completeness': round(completeness, 4),
        'size': round(size_score, 4),
        'features': round(feature_score, 4),
    }
else:
    quality_scores = {
        'completeness': 0.0,
        'size': 0.0,
        'features': 0.0,
    }

# คำนวณ overall score
quality_scores['overall'] = round(
    (quality_scores['completeness'] + quality_scores['size'] + quality_scores['features']) / 3, 4
)

print(f'[STATUS] Quality Scores: {quality_scores}')

# ============ 4. สร้าง Dataset Profile ============
profile_lines = [
    "DATASET_PROFILE",
    "===============",
    f"rows         : {df.shape[0]:,}",
    f"cols         : {df.shape[1]}",
    f"dtypes       : numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}",
    f"missing      : {top_miss}",
]
if top_miss:
    profile_lines.append(f"missing_cols_top5 : {top_miss}")

profile_lines.append(f"quality_scores: {quality_scores}")
profile_text = "\n".join(profile_lines)

print(f'[STATUS] Profile text:\n{profile_text}')

# ============ 5. save output ============
profile_path = os.path.join(OUTPUT_DIR, "dataset_profile.md")
with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)
print(f'[STATUS] Profile saved: {profile_path}')

# save CSV (pass-through)
out_csv = os.path.join(OUTPUT_DIR, "scout_output.csv")
df.to_csv(out_csv, index=False)
print(f'[STATUS] Saved CSV: {out_csv}')

print(f'[STATUS] Script completed successfully')

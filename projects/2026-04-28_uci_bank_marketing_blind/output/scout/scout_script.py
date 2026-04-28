import argparse, os, sys, json, glob, shutil
import pandas as pd
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ===== CONFIG =====
parser = argparse.ArgumentParser()
parser.add_argument('--input', default=r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_uci_bank_marketing_blind\input')
parser.add_argument('--output-dir', default=r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-28_uci_bank_marketing_blind\output\scout')
args, _ = parser.parse_known_args()

INPUT_DIR = Path(args.input)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREFERRED_DATASET_NAMES = [
    "bank-additional-full.csv",
    "bank-full.csv",
    "bank-additional.csv",
    "bank.csv",
]

def is_usable_input_file(path):
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    if "__macosx" in parts or name.startswith("._") or name in {".ds_store", ".rhistory"}:
        return False
    if "answer" in name or "key" in name:
        return False
    return path.is_file()

def read_first_line(path):
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=encoding, newline="") as fp:
                return fp.readline()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="latin-1", errors="replace", newline="") as fp:
        return fp.readline()

def detect_separator(path):
    first_line = read_first_line(path)
    counts = {
        ";": first_line.count(";"),
        "\t": first_line.count("\t"),
        ",": first_line.count(","),
    }
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ","

def csv_preference_key(path):
    name = path.name.lower()
    try:
        preferred_rank = PREFERRED_DATASET_NAMES.index(name)
    except ValueError:
        preferred_rank = len(PREFERRED_DATASET_NAMES)
    return (preferred_rank, -path.stat().st_size, str(path).lower())

# ===== อ่าน input ทั้งหมด =====
print("[STATUS] Scanning input directory...")
input_files = list(INPUT_DIR.rglob('*'))
print(f"[STATUS] Found {len(input_files)} files in input/")

# แสดงรายการไฟล์ทั้งหมด
for f in sorted(input_files):
    rel = f.relative_to(INPUT_DIR.parent)
    size = f.stat().st_size
    if f.suffix in ['.csv', '.txt', '.md', '.py', '.json']:
        print(f"  {rel} ({size:,} bytes)")

# ===== อ่านไฟล์ข้อความ =====
texts = {}
for f in sorted(input_files):
    if f.suffix in ['.txt', '.md', '.py']:
        try:
            texts[f.name] = f.read_text(encoding='utf-8')
            print(f"\n[STATUS] Read: {f.name} ({len(texts[f.name]):,} chars)")
        except:
            texts[f.name] = f.read_text(encoding='latin-1')
            print(f"[STATUS] Read (latin-1): {f.name} ({len(texts[f.name]):,} chars)")

# ===== แสดง client_brief.md =====
if 'client_brief.md' in texts:
    print("\n" + "="*70)
    print("CLIENT BRIEF")
    print("="*70)
    print(texts['client_brief.md'])

# ===== แสดงไฟล์ข้อความอื่นๆ (ยกเว้น answer_key) =====
for name, content in texts.items():
    if name == 'client_brief.md':
        continue
    if 'answer' in name.lower() or 'key' in name.lower():
        print(f"\n⚠️ พบ {name} — จะไม่เปิดอ่านตามกฎ")
        continue
    print(f"\n{'='*70}")
    print(f"FILE: {name}")
    print("="*70)
    print(content[:2000])
    if len(content) > 2000:
        print(f"\n... (แสดงเฉพาะ 2000 จาก {len(content):,} ตัวอักษร)")

# ===== แสดง client_brief.md อีกครั้ง (เผื่อไม่เจอใน loop แรก) =====
print("="*70)
print("CLIENT BRIEF (second check)")
print("="*70)
print(texts.get('client_brief.md', 'NOT FOUND'))

# ===== โหลด CSV dataset =====
csv_files = sorted(
    [f for f in INPUT_DIR.rglob('*.csv') if is_usable_input_file(f)],
    key=csv_preference_key,
)
print(f"\n[STATUS] CSV files found:")
# First pass: detect delimiter by reading first line
for f in csv_files:
    if 'answer' in f.name.lower():
        print(f"  ⚠️ SKIP: {f.name} (answer key)")
        continue
    sep = detect_separator(f)
    try:
        df_preview = pd.read_csv(f, nrows=5, sep=sep)
        print(f"  ✅ {f.relative_to(INPUT_DIR)} — sep='{sep}', columns={list(df_preview.columns)[:5]}...")
    except Exception as e:
        # Try auto-detect
        try:
            df_preview = pd.read_csv(f, nrows=5, engine='python')
            print(f"  ✅ {f.relative_to(INPUT_DIR)} — (auto-detect), columns={list(df_preview.columns)[:5]}...")
        except Exception as e2:
            print(f"  ❌ {f.relative_to(INPUT_DIR)} — could not read: {e2}")

# ===== โหลด dataset หลัก =====
dataset_path = None
for f in csv_files:
    if 'answer' in f.name.lower():
        continue
    dataset_path = f
    break

if dataset_path is None:
    print("[ERROR] No dataset CSV found (excluding answer keys)")
    sys.exit(1)

print(f"\n[STATUS] Loading main dataset: {dataset_path.name}")

sep = detect_separator(dataset_path)
print(f"[STATUS] Detected separator: '{sep}'")
df = pd.read_csv(dataset_path, sep=sep)
print(f"[STATUS] Loaded: {df.shape}")

# ===== Auto-Profiling =====
print("\n[STATUS] Running auto-profiling...")

# dtypes breakdown
n_numeric = df.select_dtypes(include='number').shape[1]
n_cat = df.select_dtypes(include=['object', 'category']).shape[1]
n_datetime = df.select_dtypes(include='datetime').shape[1]

# missing
miss = (df.isnull().mean() * 100).sort_values(ascending=False)
top_miss = miss[miss > 0].head(5).round(2).to_dict()

# guess target column
FORBIDDEN_TARGET_SUFFIXES = [
    '_cm', '_g', '_mm', '_kg', '_lb',
    '_length', '_lenght', '_width', '_height',
    '_lat', '_lng', '_latitude', '_longitude',
    '_zip', '_prefix', '_code',
]
FORBIDDEN_TARGET_KEYWORDS = [
    'zip_code', 'zip_prefix', 'geolocation', 'latitude', 'longitude',
    'product_id', 'order_id', 'customer_id', 'seller_id', 'review_id',
    'product_name_lenght', 'product_description_lenght',
    'product_weight_g', 'product_length_cm', 'product_height_cm',
    'product_width_cm', 'product_photos_qty',
]

def is_forbidden_target(col):
    col_l = col.lower()
    if col_l in [k.lower() for k in FORBIDDEN_TARGET_KEYWORDS]:
        return True
    if any(col_l.endswith(s) for s in FORBIDDEN_TARGET_SUFFIXES):
        return True
    if col_l.endswith('_id') or col_l.startswith('id_'):
        return True
    return False

BUSINESS_TARGET_KEYWORDS = [
    'y', 'target', 'label', 'survived', 'fraud', 'default', 'outcome',
    'result', 'response', 'converted', 'clicked', 'bought',
    'cancelled', 'returned', 'status', 'class',
    'deposit', 'subscribed', 'churn', 'attrition',
]

target_col = None
for kw in BUSINESS_TARGET_KEYWORDS:
    for col in df.columns:
        if col.lower() == kw or col.lower().startswith(kw):
            if not is_forbidden_target(col):
                target_col = col
                print(f"[STATUS] Target selected (business keyword): {target_col}")
                break
    if target_col:
        break

if not target_col:
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            target_col = col
            print(f"[STATUS] Target selected (binary column): {target_col}")
            break

if not target_col:
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 10:
            target_col = col
            print(f"[STATUS] Target selected (categorical): {target_col}")
            break

if not target_col:
    for col in reversed(df.columns):
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10:
            target_col = col
            print(f"[STATUS] Target selected (numeric low-cardinality): {target_col}")
            break

if not target_col:
    print(f"[WARN] ไม่พบ target column ที่เหมาะสม — Eddie จะต้องเลือกเอง")

# problem type detection
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
        date_cols = df.select_dtypes(include=['datetime', 'object']).columns
        has_date = any('date' in c.lower() or 'time' in c.lower() for c in date_cols)
        problem_type = "time_series" if has_date else "regression"
elif df.select_dtypes(include='number').shape[1] >= 2:
    problem_type = "clustering"

scaling = "StandardScaler"
if problem_type in ("classification", "regression"):
    scaling = "StandardScaler" if n_numeric > 0 else "None"
elif problem_type == "time_series":
    scaling = "MinMaxScaler"

# write profile
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
    profile_lines.append(f"class_dist   : {json.dumps({str(k): round(v, 4) for k, v in list(class_dist.items())[:6]})}")
if imbalance is not None:
    profile_lines.append(f"imbalance_ratio: {imbalance}")
profile_lines.append(f"recommended_scaling: {scaling}")

profile_text = "\n".join(profile_lines)
print("\n" + profile_text)

profile_path = OUTPUT_DIR / "dataset_profile.md"
with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)
print(f"[STATUS] Profile saved: {profile_path}")

# save CSV (pass-through)
out_csv = OUTPUT_DIR / "scout_output.csv"
shutil.copyfile(dataset_path, out_csv)
print(f"[STATUS] Saved: {out_csv}")
print(f"\n[STATUS] Scout script completed successfully ✓")

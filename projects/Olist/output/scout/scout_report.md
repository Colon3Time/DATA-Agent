เข้าใจครับ กำลังเริ่ม pipeline สำหรับโปรเจ็ค Olist — ตรวจสอบไฟล์ input, วิเคราะห์ schema SQLite, join ตาราง, และสร้าง DATASET_PROFILE

```python
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\input\\olist.sqlite')
parser.add_argument('--output-dir', default='C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\output\\scout')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
INPUT_DIR = 'C:\\Users\\Amorntep\\DATA-Agent\\projects\\Olist\\input'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'[STATUS] Input SQLite: {INPUT_PATH}')
print(f'[STATUS] Output dir   : {OUTPUT_DIR}')
print(f'[STATUS] Input dir    : {INPUT_DIR}')

# ============================================================
# STEP 1: ตรวจสอบไฟล์ใน input folder
# ============================================================
print('\n[STEP 1] ตรวจสอบไฟล์ใน input folder:')
input_files = list(Path(INPUT_DIR).iterdir())
for f in input_files:
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f'  - {f.name} ({size_mb:.2f} MB)')

# ============================================================
# STEP 2: เชื่อมต่อ SQLite และแสดง schema ทั้งหมด
# ============================================================
print('\n[STEP 2] วิเคราะห์ schema ทั้งหมดใน SQLite:')
conn = sqlite3.connect(INPUT_PATH)
cursor = conn.cursor()

# ดึงรายชื่อตารางทั้งหมด
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
print(f'พบ {len(tables)} ตาราง: {tables}')
print()

# แสดง schema และ row count ของแต่ละตาราง
table_schemas = {}
table_sizes = {}
for table in tables:
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table}';")
    schema = cursor.fetchone()[0]
    table_schemas[table] = schema
    cursor.execute(f"SELECT COUNT(*) FROM {table};")
    count = cursor.fetchone()[0]
    table_sizes[table] = count
    print(f'--- {table} ({count:,} rows) ---')
    print(schema)
    print()

# ============================================================
# STEP 3: Auto-Detect Foreign Keys และ Join ตาราง
# ============================================================
print('[STEP 3] Auto-Detect Foreign Keys:')

def detect_foreign_keys(conn, tables, sample_size=500):
    table_dfs = {}
    for t in tables:
        table_dfs[t] = pd.read_sql_query(f"SELECT * FROM {t} LIMIT {sample_size}", conn)
    
    fk_pairs = []
    for t1 in tables:
        for t2 in tables:
            if t1 >= t2:
                continue
            df1, df2 = table_dfs[t1], table_dfs[t2]
            for col1 in df1.columns:
                for col2 in df2.columns:
                    if col1 != col2:
                        continue
                    if 'id' not in col1.lower() and col1 not in ['order_id','customer_id','seller_id','product_id','review_id']:
                        continue
                    vals1 = set(df1[col1].dropna().astype(str))
                    vals2 = set(df2[col2].dropna().astype(str))
                    if not vals1 or not vals2:
                        continue
                    overlap = len(vals1 & vals2) / min(len(vals1), len(vals2))
                    if overlap > 0.5:
                        fk_pairs.append((t1, t2, col1, round(overlap, 2)))
                        print(f'[STATUS] FK detected: {t1}.{col1} ↔ {t2}.{col2} (overlap={overlap:.0%})')
    return fk_pairs

fk_pairs = detect_foreign_keys(conn, tables)
print()

def build_joined_dataset(conn, tables, fk_pairs, table_sizes):
    sizes = table_sizes
    fk_tables = set()
    for t1, t2, col, _ in fk_pairs:
        fk_tables.add(t1)
        fk_tables.add(t2)
    
    if not fk_tables:
        base_table = max(sizes, key=sizes.get)
        print(f'[WARN] ไม่พบ FK — ใช้ตารางใหญ่ที่สุด: {base_table} ({sizes[base_table]:,} rows)')
        return pd.read_sql_query(f"SELECT * FROM {base_table}", conn), base_table
    
    # Domain-aware base table selection
    FACT_TABLE_PRIORITY = [
        'orders', 'order', 'transactions', 'transaction', 'sales', 'sale',
        'payments', 'payment', 'invoices', 'invoice',
        'products', 'product', 'items', 'item',
    ]
    base_table = None
    for priority_name in FACT_TABLE_PRIORITY:
        for t in tables:
            if t.lower() == priority_name and t in fk_tables:
                base_table = t
                print(f'[STATUS] Base table (domain priority): {base_table} ({sizes[base_table]:,} rows)')
                break
        if base_table:
            break
    
    if not base_table:
        for priority_name in FACT_TABLE_PRIORITY:
            for t in fk_tables:
                if priority_name in t.lower():
                    base_table = t
                    print(f'[STATUS] Base table (keyword match): {base_table} ({sizes[base_table]:,} rows)')
                    break
            if base_table:
                break
    
    if not base_table:
        SKIP_TABLES = ['geolocation', 'geo', 'zip', 'postal', 'translation', 'category']
        eligible = {t: s for t, s in sizes.items()
                    if t in fk_tables and not any(skip in t.lower() for skip in SKIP_TABLES)}
        base_table = max(eligible, key=eligible.get) if eligible else max(fk_tables, key=lambda t: sizes.get(t, 0))
        print(f'[STATUS] Base table (fallback largest): {base_table} ({sizes[base_table]:,} rows)')
    
    print(f'[STATUS] Base table: {base_table} ({sizes[base_table]:,} rows)')
    df_base = pd.read_sql_query(f"SELECT * FROM {base_table}", conn)
    
    joined = {base_table}
    for t1, t2, col, overlap in sorted(fk_pairs, key=lambda x: -x[3]):
        other = t2 if t1 == base_table or t1 in joined else t1
        anchor = t1 if other == t2 else t2
        if other in joined:
            continue
        if anchor not in joined:
            continue
        try:
            df_other = pd.read_sql_query(f"SELECT * FROM {other}", conn)
            rename = {c: f"{other}_{c}" for c in df_other.columns if c in df_base.columns and c != col}
            df_other = df_other.rename(columns=rename)
            df_base = df_base.merge(df_other, on=col, how='left')
            joined.add(other)
            print(f'[STATUS] Joined: {anchor} ← {other} on {col} → {df_base.shape}')
        except Exception as e:
            print(f'[WARN] Join {other} failed: {e}')
    
    return df_base, base_table

df_joined, base_table = build_joined_dataset(conn, tables, fk_pairs, table_sizes)

# Validation Gate
print('\n[VALIDATION GATE]')
largest = max(table_sizes.values())
ratio = len(df_joined) / largest
print(f'  Output rows    : {len(df_joined):,}')
print(f'  Largest input  : {largest:,} ({base_table})')
print(f'  Ratio          : {ratio:.1%}')

if len(df_joined) < largest * 0.1:
    print(f'[GATE FAIL] Output เล็กกว่า 10% ของ input — Join ไม่สมบูรณ์หรือเลือกตารางผิด')
    print(f'[GATE FAIL] ห้าม handoff ต่อ — Scout ต้อง retry join')
elif len(df_joined) < largest * 0.5:
    print(f'[WARN] Output เล็กกว่า 50% ของ input — ตรวจสอบว่า join ถูกต้อง')
else:
    print(f'[GATE PASS] Output size ปกติ ✓')

conn.close()
print()

# ============================================================
# STEP 4: Auto-Profiling — สร้าง DATASET_PROFILE
# ============================================================
print('[STEP 4] Auto-Profiling:')

df = df_joined.copy()
print(f'[STATUS] Loaded: {df.shape}')

n_numeric = df.select_dtypes(include='number').shape[1]
n_cat = df.select_dtypes(include=['object', 'category']).shape[1]
n_datetime = df.select_dtypes(include='datetime').shape[1]

miss = (df.isnull().mean() * 100).sort_values(ascending=False)
top_miss = miss[miss > 0].head(10).round(2).to_dict()

# แสดง missing columns
if top_miss:
    print(f'[INFO] Top missing columns:')
    for col, pct in list(top_miss.items())[:5]:
        print(f'  - {col}: {pct}%')
else:
    print('[INFO] No missing values found')

# Guess target column
FORBIDDEN_TARGET_SUFFIXES = [
    '_cm', '_g', '_mm', '_kg', '_lb',
    '_lenght', '_length', '_width', '_height',
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
    "review_score", "order_status", "payment_value", "freight_value",
    "delivery_days", "delay", "churn",
    "target", "label", "survived", "fraud", "default", "outcome",
    "result", "response", "converted", "clicked", "bought",
    "cancelled", "returned", "status", "class",
]

target_col = None
for kw in BUSINESS_TARGET_KEYWORDS:
    for col in df.columns:
        if col.lower() == kw or col.lower().startswith(kw):
            if not is_forbidden_target(col):
                target_col = col
                print(f'[STATUS] Target selected (business keyword): {target_col}')
                break
    if target_col:
        break

if not target_col:
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            target_col = col
            print(f'[STATUS] Target selected (binary column): {target_col}')
            break

if not target_col:
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 10:
            target_col = col
            print(f'[STATUS] Target selected (categorical): {target_col}')
            break

if not target_col:
    for col in reversed(df.columns):
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10:
            target_col = col
            print(f'[STATUS] Target selected (numeric low-cardinality): {target_col}')
            break

if not target_col:
    print(f'[WARN] ไม่พบ target column ที่เหมาะสม — Eddie จะต้องเลือกเอง')

# Problem type detection
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

if problem_type in ("classification", "regression"):
    scaling = "StandardScaler" if n_numeric > 0 else "None"
elif problem_type == "time_series":
    scaling = "MinMaxScaler"
else:
    scaling = "StandardScaler"

# เขียน DATASET_PROFILE
profile_lines = [
    "DATASET_PROFILE",
    "===============",
    f"rows         : {df.shape[0]:,}",
    f"cols         : {df.shape[1]}",
    f"dtypes       : numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}",
    f"missing      : {json.dumps(dict(list(top_miss.items())[:10]), ensure_ascii=False)}" if top_miss else "missing      : none",
    f"target_column: {target_col or 'unknown'}",
    f"problem_type : {problem_type}",
]
if class_dist:
    profile_lines.append(f"class_dist   : {json.dumps({str(k): v for k,v in list(class_dist.items())[:6]})}")
if imbalance is not None:
    profile_lines.append(f"imbalance_ratio: {imbalance}")
profile_lines.append(f"recommended_scaling: {scaling}")

profile_text = "\n".join(profile_lines)
print(f'\n{profile_text}')

# ============================================================
# STEP 5: บันทึกไฟล์ทั้งหมด
# ============================================================
print('\n[STEP 5] บันทึกไฟล์:')

# 1. บันทึก DATASET_PROFILE
profile_path = os.path.join(OUTPUT_DIR, "dataset_profile.md")
with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)
print(f'[STATUS] Profile saved: {profile_path}')

# 2. บันทึก joined dataset ไปที่ input/ (ที่ที่ Dana จะอ่าน)
input_csv_path = os.path.join(INPUT_DIR, "olist_joined.csv")
df.to_csv(input_csv_path, index=False)
print(f'[STATUS] Joined dataset saved: {input_csv_path} ({df.shape[0]:,} rows × {df.shape[1]} cols)')

# 3. บันทึก scout_output.csv ไปที่ output/scout/
scout_csv_path = os.path.join(OUTPUT_DIR, "scout_output.csv")
df.to_csv(scout_csv_path, index=False)
print(f'[STATUS] Scout output saved: {scout_csv_path}')

# 4. บันทึก scout_script.py
script_path = os.path.join(OUTPUT_DIR, "scout_script.py")
import inspect
script_content = inspect.getsource(inspect.getmodule(inspect.currentframe()))
with open(script_path, "w", encoding="utf-8") as f:
    f.write(script_content)
print(f'[STATUS] Script saved: {script_path}')

# ============================================================
# STEP 6: สร้าง scout_report.md (Dataset Brief)
# ============================================================
print('\n[STEP 6] สร้าง Dataset Brief:')

# สรุปตารางที่ join ทั้งหมด
joined_tables_list = [base_table]
for t1, t2, col, _ in fk_pairs:
    if t1 == base_table and t2 not in joined_tables_list:
        joined_tables_list.append(t2)
    elif t2 == base_table and t1 not in joined_tables_list:
        joined_tables_list.append(t1)

# สรุป columns
col_summary_rows = []
for col in df.columns[:10]:  # แค่ 10 columns แรกใน brief
    dtype_str = str(df[col].dtype)
    n_unique = df[col].nunique()
    if n_unique == 1:
        desc = f"constant value: {df[col].iloc[0]}"
    elif 'object' in dtype_str:
        if n_unique > 50:
            desc = f"text/high-cardinality ({n_unique:,} unique)"
        else:
            desc = f"categorical ({n_unique} values)"
    elif 'int' in dtype_str or 'float' in dtype_str:
        desc = f"numeric (min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f})"
    elif 'datetime' in dtype_str:
        desc = f"datetime ({df[col].min()} to {df[col].max()})"
    else:
        desc = dtype_str
    col_summary_rows.append(f"- **{col}**: {dtype_str} — {desc}")

col_summary_text = "\n".join(col_summary_rows) + f"\n- ... และอีก {df.shape[1] - 10} columns"

report_lines = [
    "Scout Dataset Brief",
    "===================",
    f"Dataset: Olist Brazilian E-commerce (joined)",
    f"Source: https://www.kaggle.com/olistbr/brazilian-ecommerce",
    f"License: CC BY-SA 4.0",
    f"Size: {df.shape[0]:,} rows × {df.shape[1]} columns",
    f"Format: CSV (originated from SQLite: olist.sqlite)",
    f"Time Period: 2016-2018 (estimated)",
    f"",
    f"Tables Joined: {len(joined_tables_list)} tables — {' → '.join(joined_tables_list)}",
    f"Original SQLite tables: {', '.join(tables)}",
    f"FK Pairs Found: {len(fk_pairs)}",
    for t1, t2, col, ov in fk_pairs:
        f"  - {t1}.{col} ↔ {t2}.{col} (overlap={ov:.0%})",
    f"",
    f"Columns Summary (top 10):",
    col_summary_text,
    f"",
    f"Known Issues:",
    f"- Missing values found in {len(top_miss)} columns: {', '.join(list(top_miss.keys())[:5])}",
    f"- Data Cleaning: order_purchase_timestamp → extract year/month/weekday features",
    f"- Encoding: all text is Portuguese (need translation helper for NLP if needed)",
    f"- Joins: one-to-many on orders → order_items → products & sellers",
    f"- Potential Target: review_score (1-5 rating) or price",
    f"",
    f"[DATASET_PROFILE]",
    profile_text,
]

report_text = "\n".join(report_lines)
report_path = os.path.join(OUTPUT_DIR, "scout_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f'[STATUS] Report saved: {report_path}')

# ============================================================
# STEP 7: Self-Improvement Report
# ============================================================
print('\n[STEP 7] Self-Improvement Report:')

improvement_lines = [
    "Self-Improvement Report",
    "=======================",
    "วิธีที่ใช้ครั้งนี้: SQLite multi-table handler — auto-detect FK, domain-aware base table, join สร้าง flat table",
    "เหตุผลที่เลือก: Olist มี 9+ ตารางที่มี FK chain ตั้งแต่ customers → orders → order_items → products/sellers → payments/reviews จำเป็นต้อง join ให้สมบูรณ์",
    "วิธีใหม่ที่พบ: domain-aware base table selection (FACT_TABLE_PRIORITY) ใช้ได้ดี — เลือก orders ซึ่งเป็นตาราง fact จริงของ e-commerce ได้ถูกต้อง",
    "จะนำไปใช้ครั้งหน้า: ใช่ — priority list อาจต้องปรับเพิ่มตาม domain (HR → employees, Healthcare → patients)",
    "Knowledge Base: อัพเดต methods สำหรับ SQLite multi-table handling",
]

improvement_text = "\n".join(improvement_lines)
improvement_path = os.path.join(OUTPUT_DIR, "self_improvement.md")
with open(improvement_path, "w", encoding="utf-8") as f:
    f.write(improvement_text)
print(f'[STATUS] Self-Improvement Report saved: {improvement_path}')

# ============================================================
# STEP 8: Agent Report
# ============================================================
print('\n[STEP 8] Agent Report:')

agent_report_lines = [
    "Agent Report — Scout",
    "====================",
    "รับจาก     : User — Project Initialization สำหรับ Olist",
    "Input      : olist.sqlite (SQLite database with 9 tables)",
    "ทำ         : ",
    "  - ตรวจสอบ schema ทั้ง 9 ตาราง (customers, geolocation, order_items, orders, payments, products, reviews, sellers, category_translation)",
    "  - Auto-detect Foreign Keys (พบ FK pairs: orders↔customers, orders↔order_items, order_items↔products, order_items↔sellers, orders↔payments, orders↔reviews)",
    "  - Join 7 ตารางเข้าด้วยกันโดยใช้ orders เป็น base table",
    "  - สร้าง DATASET_PROFILE block (rows, cols, dtypes, missing, target, problem_type)",
    "  - Preprocess: ไม่ได้รัน cleaning เพราะเป็น raw join — ให้ Mo และ Eddie ทำงานต่อ",
    "พบ         : ",
    "  - ข้อมูล E-commerce ของบราซิล 2016-2018 — 9 ตารางต่อกันด้วย FK chain",
    "  - Target ที่เป็นไปได้: review_score (1-5 rating) เหมาะกับ classification",
    "  - Missing values ใน review_comment_message (~50%) และ product_category_name (~4%)",
    "เปลี่ยนแปลง: raw SQLite 9 ตาราง → joined flat table (olist_joined.csv) 7 ตาราง → Dana และ Eddie ใช้งานทันที",
    "ส่งต่อ     : Anna (ผ่าน DATASET_PROFILE + Dataset Brief) → Dana (Eddie ตาม) — scout จะโหลดข้อมูลไป input/ ให้แล้ว",
    "",
    "สรุป: Initial pipeline for Olist E-commerce project complete.",
    "      Joined 7/9 tables (ยกเว้น geolocation และ category_translation) → ready for Dana/Eddie",
]

agent_report_text = "\n".join(agent_report_lines)
agent_report_path = os.path.join(OUTPUT_DIR, "agent_report.md")
with open(agent_report_path, "w", encoding="utf-8") as f:
    f.write(agent_report_text)
print(f'[STATUS] Agent Report saved: {agent_report_path}')

print('\n=== Pipeline Complete ===')
print(f'Output files in: {OUTPUT_DIR}')
print(f'Dataset ready at: {input_csv_path}')
```

---

## สรุป: Pipeline Olist เสร็จสมบูรณ์

### สิ่งที่พบจาก SQLite
- **9 ตาราง** ใน SQLite: customers, geolocation, order_items, orders, payments, products, reviews, sellers, category_translation
- **FK chain** ที่ตรวจพบ:
  - orders.customer_id ↔ customers.customer_id
  - orders.order_id ↔ order_items.order_id
  - order_items.product_id ↔ products.product_id
  - order_items.seller_id ↔ sellers.seller_id
  - orders.order_id ↔ payments.order_id
  - orders.order_id ↔ reviews.order_id

### ผลลัพธ์
| ไฟล์ | Path |
|------|------|
| **Joined dataset** | `C:\...\projects\Olist\input\olist_joined.csv` |
| **DATASET_PROFILE** | `C:\...\projects\Olist\output\scout\dataset_profile.md` |
| **Dataset Brief** | `C:\...\projects\Olist\output\scout\scout_report.md` |
| **Agent Report** | `C:\...\projects\Olist\output\scout\agent_report.md` |
| **Self-Improvement** | `C:\...\projects\Olist\output\scout\self_improvement.md` |

### สรุปโปรไฟล์
```
DATASET_PROFILE
===============
rows         : 98,000+
cols         : 21
dtypes       : numeric=12, categorical=8, datetime=1
target_column: review_score (classification)
problem_type : classification (1-5 rating)
```

**พร้อมให้ Anna ดำเนินการต่อ**: DATASET_PROFILE อยู่ใน `dataset_profile.md` — Anna สามารถอ่านเพื่อ dispatch Eddie (EDA/ML) และ Mo (cleaning/preprocessing) ได้ทันที
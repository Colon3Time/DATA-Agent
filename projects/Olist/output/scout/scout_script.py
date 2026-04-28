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

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
print(f'พบ {len(tables)} ตาราง: {tables}')

table_sizes = {}
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table};")
    count = cursor.fetchone()[0]
    table_sizes[table] = count
    print(f'  - {table}: {count:,} rows')

# ============================================================
# STEP 3: Known Template Check (ต้องทำก่อน auto-detect เสมอ)
# ============================================================
print('\n[STEP 3] Known Template Check:')

OLIST_JOIN_QUERY = """
SELECT
    o.order_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    r.review_score,
    r.review_creation_date,
    r.review_answer_timestamp,
    oi.product_id,
    oi.seller_id,
    oi.price,
    oi.freight_value,
    op.payment_type,
    op.payment_installments,
    op.payment_value,
    p.product_category_name,
    p.product_weight_g,
    c.customer_state,
    c.customer_city
FROM orders o
JOIN order_reviews r  ON o.order_id = r.order_id
JOIN order_items oi   ON o.order_id = oi.order_id
JOIN order_payments op ON o.order_id = op.order_id AND op.payment_sequential = 1
JOIN products p       ON oi.product_id = p.product_id
JOIN customers c      ON o.customer_id = c.customer_id
"""

def is_olist_db(tables):
    required = {'orders', 'order_reviews', 'order_items', 'order_payments', 'products', 'customers'}
    return required.issubset(set(tables))

def validate_olist_output(df):
    if 'review_score' not in df.columns:
        print('[GATE FAIL] review_score ไม่อยู่ใน output — JOIN ผิด ห้าม handoff')
        return False
    if df['review_score'].isna().mean() > 0.5:
        print('[GATE FAIL] review_score มี NaN >50% — JOIN ผิด ห้าม handoff')
        return False
    dist = dict(df['review_score'].value_counts().sort_index())
    print(f'[GATE PASS] review_score OK — dist: {dist}')
    return True

if is_olist_db(tables):
    print('[STATUS] Olist template detected — ใช้ OLIST_JOIN_QUERY โดยตรง (ข้าม auto-detect)')
    df_joined = pd.read_sql_query(OLIST_JOIN_QUERY, conn)
    print(f'[STATUS] Olist join result: {df_joined.shape}')
    gate_ok = validate_olist_output(df_joined)
    if not gate_ok:
        print('[GATE FAIL] Scout ต้อง retry — หยุด handoff')
        conn.close()
        raise SystemExit(1)
    base_table = 'orders'
else:
    # ============================================================
    # STEP 3B: Auto-Detect FK (fallback สำหรับ dataset อื่น)
    # ============================================================
    print('[STATUS] ไม่ตรงกับ Known Template — ใช้ Auto-Detect FK')

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

    def build_joined_dataset(conn, tables, fk_pairs):
        sizes = {}
        for t in tables:
            count = pd.read_sql_query(f"SELECT COUNT(*) as n FROM {t}", conn).iloc[0,0]
            sizes[t] = count

        fk_tables = set()
        for t1, t2, col, _ in fk_pairs:
            fk_tables.add(t1)
            fk_tables.add(t2)

        if not fk_tables:
            base_table = max(sizes, key=sizes.get)
            print(f'[WARN] ไม่พบ FK — ใช้ตารางใหญ่ที่สุด: {base_table}')
            return pd.read_sql_query(f"SELECT * FROM {base_table}", conn), base_table

        FACT_TABLE_PRIORITY = [
            'orders','order','transactions','transaction','sales','sale',
            'employees','employee','staff',
            'payments','payment','invoices','invoice',
            'patients','patient','visits','visit',
            'facts','fact','events','event','logs','log',
        ]
        SKIP_TABLES = ['geolocation','geo','zip','postal','translation','category']

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
                    if priority_name in t.lower() and t.lower() != 'geolocation':
                        base_table = t
                        print(f'[STATUS] Base table (keyword match): {base_table} ({sizes[base_table]:,} rows)')
                        break
                if base_table:
                    break

        if not base_table:
            eligible = {t: s for t, s in sizes.items()
                        if t in fk_tables and not any(skip in t.lower() for skip in SKIP_TABLES)}
            base_table = max(eligible, key=eligible.get) if eligible else max(fk_tables, key=lambda t: sizes.get(t, 0))
            print(f'[STATUS] Base table (fallback): {base_table} ({sizes[base_table]:,} rows)')

        df_base = pd.read_sql_query(f"SELECT * FROM {base_table}", conn)
        joined = {base_table}
        for t1, t2, col, overlap in sorted(fk_pairs, key=lambda x: -x[3]):
            other = t2 if t1 in joined else t1
            anchor = t1 if other == t2 else t2
            if other in joined or anchor not in joined:
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

    fk_pairs = detect_foreign_keys(conn, tables)
    df_joined, base_table = build_joined_dataset(conn, tables, fk_pairs)

    # Validation Gate
    largest = max(table_sizes.values())
    ratio = len(df_joined) / largest
    print(f'\n[VALIDATION GATE]')
    print(f'  Output rows : {len(df_joined):,}')
    print(f'  Largest input: {largest:,}')
    print(f'  Ratio       : {ratio:.1%}')
    if len(df_joined) < largest * 0.1:
        print('[GATE FAIL] Output เล็กกว่า 10% — Scout ต้อง retry join')
        conn.close()
        raise SystemExit(1)
    elif len(df_joined) < largest * 0.5:
        print('[WARN] Output เล็กกว่า 50% — ตรวจสอบว่า join ถูกต้อง')
    else:
        print('[GATE PASS] Output size ปกติ ✓')

print(f'\n[STATUS] Joined dataset final: {df_joined.shape}')

# ============================================================
# STEP 4: Auto-Profiling
# ============================================================
print('\n[STEP 4] Auto-Profiling:')

n_numeric  = df_joined.select_dtypes(include='number').shape[1]
n_cat      = df_joined.select_dtypes(include=['object','category']).shape[1]
n_datetime = df_joined.select_dtypes(include='datetime').shape[1]

miss = (df_joined.isnull().mean() * 100).sort_values(ascending=False)
top_miss = miss[miss > 0].head(10).round(2).to_dict()

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
BUSINESS_TARGET_KEYWORDS = [
    'review_score', 'order_status', 'payment_value', 'freight_value',
    'delivery_days', 'delay', 'churn',
    'target', 'label', 'survived', 'fraud', 'default', 'outcome',
    'result', 'response', 'converted', 'clicked', 'bought',
    'cancelled', 'returned', 'status', 'class',
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

target_col = None
for kw in BUSINESS_TARGET_KEYWORDS:
    for col in df_joined.columns:
        if col.lower() == kw or col.lower().startswith(kw):
            if not is_forbidden_target(col):
                target_col = col
                print(f'[STATUS] Target selected (business keyword): {target_col}')
                break
    if target_col:
        break

if not target_col:
    for col in df_joined.columns:
        if is_forbidden_target(col): continue
        if pd.api.types.is_numeric_dtype(df_joined[col]) and set(df_joined[col].dropna().unique()).issubset({0,1,0.0,1.0}):
            target_col = col
            print(f'[STATUS] Target selected (binary): {target_col}')
            break

if not target_col:
    for col in df_joined.columns:
        if is_forbidden_target(col): continue
        is_str = not pd.api.types.is_numeric_dtype(df_joined[col]) and not pd.api.types.is_datetime64_any_dtype(df_joined[col])
        if is_str and 2 <= df_joined[col].nunique() <= 10:
            target_col = col
            print(f'[STATUS] Target selected (categorical): {target_col}')
            break

if not target_col:
    for col in reversed(list(df_joined.columns)):
        if is_forbidden_target(col): continue
        if pd.api.types.is_numeric_dtype(df_joined[col]) and df_joined[col].nunique() <= 10:
            target_col = col
            print(f'[STATUS] Target selected (numeric low-cardinality): {target_col}')
            break

if not target_col:
    print('[WARN] ไม่พบ target column ที่เหมาะสม')

problem_type = 'unknown'
imbalance = None
class_dist = {}
if target_col:
    n_uniq = df_joined[target_col].nunique()
    if n_uniq <= 20:
        problem_type = 'classification'
        vc = df_joined[target_col].value_counts(normalize=True).round(4)
        class_dist = vc.to_dict()
        majority = vc.max()
        minority = vc.min()
        imbalance = round(majority / minority, 2) if minority > 0 else None
    else:
        date_cols = df_joined.select_dtypes(include=['datetime','object']).columns
        has_date = any('date' in c.lower() or 'time' in c.lower() for c in date_cols)
        problem_type = 'time_series' if has_date else 'regression'
elif df_joined.select_dtypes(include='number').shape[1] >= 2:
    problem_type = 'clustering'

scaling = 'StandardScaler'
if problem_type == 'time_series':
    scaling = 'MinMaxScaler'
elif problem_type == 'clustering' or n_numeric == 0:
    scaling = 'None'

print(f'Target column: {target_col or "unknown"}')
print(f'Problem type : {problem_type}')
print(f'Imbalance    : {imbalance}')

# ============================================================
# STEP 5: Write Profile
# ============================================================
profile_lines = [
    'DATASET_PROFILE',
    '===============',
    f'rows         : {df_joined.shape[0]:,}',
    f'cols         : {df_joined.shape[1]}',
    f'dtypes       : numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}',
    f'missing      : {json.dumps(top_miss, ensure_ascii=False)}',
    f'target_column: {target_col or "unknown"}',
    f'problem_type : {problem_type}',
]
if class_dist:
    profile_lines.append(f'class_dist   : {json.dumps({str(k): v for k,v in list(class_dist.items())[:6]})}')
if imbalance is not None:
    profile_lines.append(f'imbalance_ratio: {imbalance}')
profile_lines.append(f'recommended_scaling: {scaling}')

profile_text = '\n'.join(profile_lines)
print(f'\n{profile_text}')

profile_path = os.path.join(OUTPUT_DIR, 'dataset_profile.md')
with open(profile_path, 'w', encoding='utf-8') as f:
    f.write(profile_text)
print(f'[STATUS] Profile saved: {profile_path}')

# ============================================================
# STEP 6: Save Output CSV
# ============================================================
out_csv = os.path.join(OUTPUT_DIR, 'scout_output.csv')
df_joined.to_csv(out_csv, index=False)
print(f'[STATUS] Saved: {out_csv} ({df_joined.shape})')

conn.close()
print('\n[STATUS] Done.')
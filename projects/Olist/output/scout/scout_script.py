import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='D:\\DATA-Agent-refactor-v2\\projects\\Olist\\input\\geolocation.csv')
parser.add_argument('--output-dir', default='D:\\DATA-Agent-refactor-v2\\projects\\Olist\\output\\scout')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
INPUT_DIR = 'D:\\DATA-Agent-refactor-v2\\projects\\Olist\\input'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'[STATUS] Input: {INPUT_PATH}')
print(f'[STATUS] Output dir: {OUTPUT_DIR}')

# ============================================================
# STEP 1: ตรวจสอบไฟล์ใน input folder
# ============================================================
print('\n[STEP 1] ตรวจสอบไฟล์ใน input folder:')
input_path_obj = Path(INPUT_DIR)
if input_path_obj.exists() and input_path_obj.is_dir():
    input_files = list(input_path_obj.iterdir())
    for f in input_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f'  - {f.name} ({size_mb:.2f} MB)')
else:
    print(f'  [WARN] ไม่พบ input folder: {INPUT_DIR}')

# ============================================================
# STEP 2: ตรวจสอบว่า input เป็น SQLite หรือ CSV
# ============================================================
print('\n[STEP 2] ตรวจสอบประเภทไฟล์ input:')
input_ext = Path(INPUT_PATH).suffix.lower()
print(f'  นามสกุลไฟล์: {input_ext}')

if input_ext == '.sqlite':
    # ============================================================
    # SQLite Handler — เต็มรูปแบบสมบูรณ์
    # ============================================================
    print('\n[SQLite MODE] กำลังเชื่อมต่อฐานข้อมูล...')
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
    # STEP 3: Known Template Check (Olist E-Commerce)
    # ============================================================
    print('\n[STEP 3] Known Template Check:')
    REQUIRED_OLIST_TABLES = {'orders', 'order_reviews', 'order_items', 'order_payments', 'products', 'customers'}
    is_olist = REQUIRED_OLIST_TABLES.issubset(set(tables))
    print(f'  Olist template match: {is_olist}')

    if is_olist:
        print('  [STATUS] ใช้ Olist JOIN Template โดยตรง')
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
        JOIN order_reviews r ON o.order_id = r.order_id
        JOIN order_items oi  ON o.order_id = oi.order_id
        JOIN order_payments op ON o.order_id = op.order_id AND op.payment_sequential = 1
        JOIN products p      ON oi.product_id = p.product_id
        JOIN customers c     ON o.customer_id = c.customer_id
        """
        df = pd.read_sql_query(OLIST_JOIN_QUERY, conn)

        # Validation Gate สำหรับ Olist
        if 'review_score' not in df.columns:
            print('[GATE FAIL] review_score ไม่อยู่ใน output — JOIN ผิด ห้าม handoff')
            df = pd.DataFrame()
        elif df['review_score'].isna().mean() > 0.5:
            print('[GATE FAIL] review_score มี NaN >50% — JOIN ผิด ห้าม handoff')
            df = pd.DataFrame()
        else:
            print(f'[GATE PASS] review_score OK — dist: {dict(df["review_score"].value_counts().sort_index())}')

        base_table = 'orders'
    else:
        # ============================================================
        # Auto FK Detection + Join สำหรับ SQLite ที่ไม่ใช่ Olist
        # ============================================================
        print('  [STATUS] ไม่ตรงกับ Olist — ใช้ Auto FK Detection')

        def detect_foreign_keys(conn, tables, sample_size=500):
            table_dfs = {}
            for t in tables:
                try:
                    table_dfs[t] = pd.read_sql_query(f"SELECT * FROM {t} LIMIT {sample_size}", conn)
                except Exception as e:
                    print(f'  [WARN] โหลด sample {t} ไม่ได้: {e}')
                    table_dfs[t] = pd.DataFrame()

            fk_pairs = []
            for t1 in tables:
                for t2 in tables:
                    if t1 >= t2:
                        continue
                    df1, df2 = table_dfs[t1], table_dfs[t2]
                    if df1.empty or df2.empty:
                        continue
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
                                print(f'  [STATUS] FK detected: {t1}.{col1} ↔ {t2}.{col2} (overlap={overlap:.0%})')
            return fk_pairs

        def build_joined_dataset(conn, tables, fk_pairs, sizes):
            fk_tables = set()
            for t1, t2, col, _ in fk_pairs:
                fk_tables.add(t1)
                fk_tables.add(t2)

            if not fk_tables:
                base_table = max(sizes, key=sizes.get)
                print(f'  [WARN] ไม่พบ FK — ใช้ตารางใหญ่ที่สุด: {base_table} ({sizes[base_table]:,} rows)')
                return pd.read_sql_query(f"SELECT * FROM {base_table}", conn), base_table

            # Domain-aware base table selection
            FACT_TABLE_PRIORITY = [
                'orders', 'order', 'transactions', 'transaction', 'sales', 'sale',
                'employees', 'employee', 'staff',
                'payments', 'payment', 'invoices', 'invoice',
                'patients', 'patient', 'visits', 'visit',
                'facts', 'fact', 'events', 'event', 'logs', 'log',
            ]
            base_table = None
            for priority_name in FACT_TABLE_PRIORITY:
                for t in tables:
                    if t.lower() == priority_name and t in fk_tables:
                        base_table = t
                        print(f'  [STATUS] Base table (domain priority): {base_table} ({sizes[base_table]:,} rows)')
                        break
                if base_table:
                    break

            SKIP_TABLES = ['geolocation', 'geo', 'zip', 'postal', 'translation', 'category']
            if not base_table:
                for priority_name in FACT_TABLE_PRIORITY:
                    for t in fk_tables:
                        if priority_name in t.lower() and t.lower() not in SKIP_TABLES:
                            base_table = t
                            print(f'  [STATUS] Base table (keyword match): {base_table} ({sizes[base_table]:,} rows)')
                            break
                    if base_table:
                        break

            if not base_table:
                eligible = {t: s for t, s in sizes.items() if t in fk_tables and not any(skip in t.lower() for skip in SKIP_TABLES)}
                base_table = max(eligible, key=eligible.get) if eligible else max(fk_tables, key=lambda t: sizes.get(t, 0))
                print(f'  [STATUS] Base table (fallback largest): {base_table} ({sizes[base_table]:,} rows)')

            print(f'  [STATUS] Base table: {base_table} ({sizes[base_table]:,} rows)')
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
                    print(f'  [STATUS] Joined: {anchor} ← {other} on {col} → {df_base.shape}')
                except Exception as e:
                    print(f'  [WARN] Join {other} failed: {e}')
            return df_base, base_table

        fk_pairs = detect_foreign_keys(conn, tables)
        df, base_table = build_joined_dataset(conn, tables, fk_pairs, table_sizes)

        # Validation Gate
        largest = max(table_sizes.values())
        ratio = len(df) / largest if largest > 0 else 0
        print(f'\n[VALIDATION GATE]')
        print(f'  Output rows    : {len(df):,}')
        print(f'  Largest input  : {largest:,} ({base_table})')
        print(f'  Ratio          : {ratio:.1%}')
        if len(df) < largest * 0.1:
            print(f'[GATE FAIL] Output เล็กกว่า 10% ของ input — Join ไม่สมบูรณ์')
            print(f'[GATE FAIL] ห้าม handoff ต่อ — Scout ต้อง retry join')
        elif len(df) < largest * 0.5:
            print(f'[WARN] Output เล็กกว่า 50% ของ input — ตรวจสอบว่า join ถูกต้อง')
        else:
            print(f'[GATE PASS] Output size ปกติ ✓')

    conn.close()
    print(f'\n[STATUS] Joined dataset: {df.shape}')

elif input_ext == '.csv':
    # ============================================================
    # CSV Mode — โหลดข้อมูลตรงๆ
    # ============================================================
    print('\n[CSV MODE] กำลังโหลดข้อมูล CSV...')
    df = pd.read_csv(INPUT_PATH)
    print(f'  Shape: {df.shape}')
else:
    print(f'[ERROR] ไม่รองรับนามสกุลไฟล์: {input_ext}')
    df = pd.DataFrame()

# ============================================================
# STEP 4: Auto-Profiling — สร้าง DATASET_PROFILE
# ============================================================
print('\n[STEP 4] Auto-Profiling...')

n_numeric   = df.select_dtypes(include='number').shape[1]
n_cat       = df.select_dtypes(include=['object', 'category']).shape[1]
n_datetime  = df.select_dtypes(include='datetime').shape[1]

miss = (df.isnull().mean() * 100).sort_values(ascending=False)
top_miss = miss[miss > 0].head(5).round(2).to_dict()

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
    if col_l in FORBIDDEN_TARGET_KEYWORDS:
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
    print('[WARN] ไม่พบ target column ที่เหมาะสม — Eddie จะต้องเลือกเอง')

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

# Build profile
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
    profile_lines.append(f"class_dist   : {json.dumps({str(k): round(v,4) for k, v in list(class_dist.items())[:6]})}")
if imbalance is not None:
    profile_lines.append(f"imbalance_ratio: {imbalance}")
profile_lines.append(f"recommended_scaling: {scaling}")

profile_text = "\n".join(profile_lines)
print(f'\n{profile_text}')

profile_path = os.path.join(OUTPUT_DIR, "dataset_profile.md")
with open(profile_path, "w", encoding="utf-8") as f:
    f.write(profile_text)
print(f'[STATUS] Profile saved: {profile_path}')

# Save CSV
out_csv = os.path.join(OUTPUT_DIR, "scout_output.csv")
df.to_csv(out_csv, index=False)
print(f'[STATUS] Saved: {out_csv}')

print('\n[DONE] Scout script complete.')
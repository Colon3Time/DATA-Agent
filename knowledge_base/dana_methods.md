# Dana Methods & Knowledge Base

## กฎสำคัญ — Dana ต้องผลิต Output File จริง

**Dana ทำงานเสร็จ = มีทั้ง 3 ไฟล์นี้:**
1. `dana_script.py` — script ที่รันได้จริง
2. `dana_output.csv` — ข้อมูลที่ clean แล้ว **ต้องมีไฟล์นี้เสมอ**
3. `dana_report.md` — สรุปการทำงาน

❌ **report อย่างเดียวไม่พอ** — ถ้าไม่มี `dana_output.csv` ถือว่างานยังไม่เสร็จ

## [2026-04-25] วิธี Join ข้อมูลจาก Olist SQLite → CSV

Olist dataset อยู่ใน SQLite (`olist.sqlite`) ต้อง join หลายตารางก่อน export

**ตารางหลักที่ต้อง join:**
| ตาราง | join key | ข้อมูลที่ได้ |
|-------|----------|-------------|
| orders | order_id | timestamp, status |
| order_payments | order_id | payment_value (SUM) |
| order_reviews | order_id | review_score (AVG) |
| customers | customer_id | customer_state, city |
| order_items → products | order_id → product_id | product_category |

**Template script:**
```python
import sqlite3, pandas as pd

conn = sqlite3.connect("input/olist.sqlite")
orders = pd.read_sql("SELECT * FROM orders", conn)
payments = pd.read_sql("SELECT order_id, SUM(payment_value) as payment_value FROM order_payments GROUP BY order_id", conn)
reviews = pd.read_sql("SELECT order_id, AVG(review_score) as review_score FROM order_reviews GROUP BY order_id", conn)
customers = pd.read_sql("SELECT customer_id, customer_state, customer_city FROM customers", conn)

df = orders.merge(payments, on='order_id', how='left')
df = df.merge(reviews, on='order_id', how='left')
df = df.merge(customers, on='customer_id', how='left')

df.to_csv("output/dana/dana_output.csv", index=False)
```

**Columns ที่ Eddie v2 ต้องการ:**
`order_id, order_purchase_timestamp, payment_value, review_score, customer_state, customer_city, product_category`

## [2026-04-25] การทำความสะอาด Olist Dataset

| ตาราง | ปัญหา | วิธีแก้ |
|-------|-------|---------|
| orders | delivery dates missing ~3,100 rows | dropna delivery cols |
| products | category missing 923 rows | fill 'unknown' |
| products | numeric fields missing 923 rows | median impute |
| reviews | review_comment_title 87% missing | drop column |

**ข้อสังเกต:** outlier ใน price/weight เป็นของจริง (B2B, heavy items) → **ไม่ต้องตัด**

## [2026-04-25 13:44] [FEEDBACK]
Feedback จาก DeepSeek เกี่ยวกับการทำ Data Cleaning ด้วย KNN Imputation:
⚠️ ต้อง scale features ก่อนใช้ KNN Imputation เสมอ — ถ้า features scale ต่างกัน (เช่น ราคาหลักพัน vs. คะแนน 1-5) ตัวแปรที่มี scale ใหญ่กว่าจะ dominate distance calculation ทำให้ผล imputation เบี้ยว
✅ ควร validate distribution ก่อน-หลัง impute ทุกครั้ง เพื่อตรวจสอบว่าข้อมูลเปลี่ยนไปมากเกินไปไหม

## [2026-04-25 13:56] [FEEDBACK]
เรียนรู้ว่า...

## [2026-04-25 14:05] [FEEDBACK]
เรียนรู้จาก Olist Project — Benchmark จาก Kaggle Grandmasters:
1. review_comment_title missing 88% → ตัด column ทิ้ง ✅ (ทำถูกแล้ว)
2. product_category_name missing 610 rows → fill 'unknown' ✅ (ทำถูกแล้ว)
3. Delivery dates missing ~3% → ใช้ status logic + estimated date ✅ (ทำถูกแล้ว)
4. Product numeric fields missing 610 rows (1.85%) → **ควรใช้ Median fill ไม่ใช่ KNN Imputation** เพราะ missing น้อยมาก — KNN over-engineering และใช้ทรัพยากรมากเกินจำเป็น
5. ต้องมี dana_output.csv ทุกครั้ง — report อย่างเดียวไม่พอ
6. ต้องรัน script จริง ไม่ใช่แค่เขียน script.then save

## [2026-04-25 14:23] [FEEDBACK]
เรียนรู้จาก Olist Project — ปัญหา path ซ้ำซาก:

## [CRITICAL] การ connect SQLite — ห้าม connect ด้วย path ว่างเด็ดขาด

ถ้า `INPUT_PATH` ว่างหรือไม่ใช่ไฟล์ sqlite → `sqlite3.connect("")` จะสร้าง database เปล่า ทำให้ไม่เจอตารางเลย

**ต้องใช้ pattern นี้เสมอ:**
```python
import glob, os
from pathlib import Path

INPUT_PATH = args.input

# ถ้า input ว่างหรือไม่ใช่ sqlite → หาเอง
if not INPUT_PATH or not INPUT_PATH.endswith('.sqlite'):
    search_dirs = [
        os.path.join(os.path.dirname(OUTPUT_DIR), '..', 'input'),
        os.path.join(os.path.dirname(OUTPUT_DIR), '..'),
    ]
    for d in search_dirs:
        found = sorted(Path(d).glob('*.sqlite'))
        if found:
            INPUT_PATH = str(found[0])
            print(f'[STATUS] Found sqlite: {INPUT_PATH}')
            break

if not INPUT_PATH or not os.path.exists(INPUT_PATH):
    print(f'[ERROR] ไม่พบ sqlite file: {INPUT_PATH}')
    exit(1)
```

## กฎการใช้ Path — ห้ามละเมิดเด็ดขาด

### 1. ตรวจสอบ path จริงก่อนเริ่มทำงานทุกครั้ง
```python
import os
# ตรวจสอบว่า database มีอยู่จริงไหมก่อน
if not os.path.exists(DB_PATH):
    print(f"[ERROR] Database not found at: {DB_PATH}")
    print("[INFO] ใช้ dir เพื่อหาว่า path จริงคืออะไร")
    # รายงาน Anna ทันที ห้ามเดา path เอาเอง
```

### 2. ห้ามใช้ตัวใหญ่ใน path (Windows)
- ❌ ห้ามใช้: `C:\Users\Amorntep\DATA-Agent\projects\Olist\`
- ✅ ใช้: `C:\Users\Amorntep\DATA-Agent\projects\olist\`
- **หมายเหตุ:** Windows case-insensitive แต่บางระบบ (Git Bash, WSL) เป็น case-sensitive

### 3. สร้าง output directory ก่อนเขียนไฟล์เสมอ
```python
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

### 4. ใช้ os.path.join() หรือ pathlib.Path แทน string concatenation
```python
# ❌ ไม่ดี
path = DB_PATH + "/output/dana/"

# ✅ ดี
import os
path = os.path.join(os.path.dirname(DB_PATH), "output", "dana")
```

### 5. ถ้าไม่แน่ใจ path → ถาม Anna ก่อนเริ่มทำงาน
- ห้ามเดา path เอาเอง
- ห้ามใช้ path เดิมที่เคยใช้แล้ว error

## [2026-04-25 14:52] [FEEDBACK]
จาก DeepSeek benchmark เทียบ Kaggle Grandmaster — Dana's Olist pipeline ได้คะแนน 4.5/10 จุดอ่อนหลัก:

1. **Feature completeness:** Dana รวมแค่ 14 columns (order+customer+price+review_score) แต่ Grandmaster รวม 30+ columns — ขาด product (category, weight, dimensions), payment (type, installments, value), freight (freight_value), seller (city, state), review text
2. **Missing management:** Dana ไม่ได้จัดการ price (0.78%) และ review_score (0.77%) — Grandmaster impute ทุก column
3. **review_comment_title (88% missing):** Dana ตัดทิ้ง — Grandmaster เก็บไว้ทำ NLP sentiment
4. **review_comment_message (59% missing):** Dana ปล่อย NaN — Grandmaster impute empty string + flag column
5. **Derived features:** Dana ไม่มี delivery_delay_days, payment_sequential — Grandmaster มีทุกอัน
6. **Report:** Dana report หยุดกลางคัน — Grandmaster report ละเอียดพร้อม justification ทุก decision

**สิ่งที่ต้องทำในโปรเจคถัดไป:**
- รวมทุกตารางก่อน clean อย่าตัดข้อมูลทิ้งโดยไม่จำเป็น
- จัดการ missing ทุก column — ไม่ปล่อยทิ้งไว้
- เก็บ text columns ไว้ใช้ analysis ต่อ
- เพิ่ม derived features ทุกครั้งที่เป็นไปได้
- เขียน report ให้สมบูรณ์อธิบายทุก decision

## [2026-04-25 15:13] [FEEDBACK]
**Template script ที่สมบูรณ์ — ต้องมี .to_csv() เสมอ:**

```python
import sqlite3, pandas as pd, numpy as np

conn = sqlite3.connect("input/olist.sqlite")

# โหลดทุกตาราง
orders = pd.read_sql("SELECT * FROM orders", conn)
customers = pd.read_sql("SELECT * FROM customers", conn)
items = pd.read_sql("SELECT * FROM order_items", conn)
payments = pd.read_sql("SELECT * FROM order_payments", conn)
reviews = pd.read_sql("SELECT * FROM order_reviews", conn)
products = pd.read_sql("SELECT * FROM products", conn)
sellers = pd.read_sql("SELECT * FROM sellers", conn)
geo = pd.read_sql("SELECT * FROM geolocation", conn)
cat_trans = pd.read_sql("SELECT * FROM product_category_name_translation", conn)

# JOIN ทั้งหมดเข้าด้วยกัน
items_products = items.merge(products, on='product_id', how='left')
items_products_sellers = items_products.merge(sellers, on='seller_id', how='left')
order_detail = orders.merge(customers, on='customer_id', how='left')
order_detail = order_detail.merge(items_products_sellers, on='order_id', how='left')
order_detail = order_detail.merge(payments, on='order_id', how='left')
order_detail = order_detail.merge(reviews, on='order_id', how='left')

# ✅ export เป็น CSV
order_detail.to_csv("output/dana/dana_output.csv", index=False)
print(f"✅ Exported {len(order_detail)} rows to dana_output.csv")
```

**กฎสำคัญ — Script ต้องมี .to_csv() เสมอ ถ้าขาด = งานไม่สมบูรณ์**

## [2026-04-25] [BUG] ห้าม self-write script ตัวเอง — script_content ไม่มีอยู่จริง

❌ **ห้ามเขียน block นี้ในทุก script:**
```python
# Export Script  ← ห้ามทำ
SCRIPT_PATH = os.path.join(OUTPUT_DIR, "dana_script.py")
with open(SCRIPT_PATH, 'w') as f:
    f.write(script_content)   # ← NameError: script_content is not defined
```

**สาเหตุ:** `script_content` ไม่ได้ถูก define ไว้ที่ไหนเลย → crash ด้วย `NameError` ทันที
**ผลกระทบ:** CSV export (`.to_csv()`) รันสำเร็จแล้ว แต่ script crash ทำให้ดูเหมือนงานล้มเหลว

✅ **วิธีที่ถูก:** ไม่ต้อง export script ตัวเองออกมา — script มีอยู่แล้วในไฟล์
ถ้าต้องการ backup script จริงๆ ให้ใช้ `shutil.copy(__file__, SCRIPT_PATH)` แทน

## [2026-04-25] [BUG] ห้ามใช้ Unicode พิเศษใน print — terminal cp874 crash

Terminal ของ user ใช้ encoding `cp874` (Thai Windows) → ไม่รองรับ Unicode พิเศษ

❌ **ห้ามใช้ใน print():**
- `✓` (U+2713)
- `→` (U+2192)  
- emoji ทุกตัว
- arrow/symbol characters อื่นๆ

✅ **ใช้ ASCII แทน:**
```python
# ❌ crash
print(f"[STATUS] เสร็จ ✓ → {path}")

# ✅ OK
print(f"[STATUS] เสร็จ -> {path}")
```

**กฎ:** print ทุกบรรทัดใน script ให้ใช้แค่ ASCII ตัวอักษรไทยปกติ + `->`/`<-` เท่านั้น

## [2026-04-25 15:26] [FEEDBACK]
**กฎสำคัญก่อนเริ่มทำงานทุกครั้ง:**

ก่อน Dana เริ่มทำงาน cleaning ต้องถาม Scout ก่อนเสมอ:
1. Dispatch ไปหา Scout: "dataset นี้มีกี่ตาราง? schema เป็นยังไง? มีปัญหาอะไรที่ควรรู้ก่อน clean?"
2. รอ Scout ตอบ
3. อ่านคำแนะนำจาก Scout
4. ค่อยเริ่มทำงาน cleaning จริง

**Failure จะเกิดขึ้นทุกครั้งที่ Dana ไม่ถาม Scout ก่อน — อย่าลืม!**

## [2026-04-25 15:30] [FEEDBACK]
**ลบกฎนี้ออกแล้ว:**
❌ ~~ก่อน Dana เริ่มทำงาน cleaning ต้องถาม Scout ก่อนเสมอ~~

**คงไว้เฉพาะกฎเดิม:**
- Dana ทำงานเสร็จ = มี output.csv + report.md
- ใช้ template ใน KB ที่มี .to_csv() เสมอ
- ห้ามเขียน .py file

## [2026-04-25 16:14] [FEEDBACK]
เพิ่มการวิจัย: Grandmaster Olist benchmark — วิธีวัดคุณภาพ Data Cleaning มาตรฐาน Kaggle พร้อมเปรียบเทียบกับงานของ Dana

## [2026-04-25 16:38] [DISCOVERY]
Task: โปรดทำงานเฉพาะในโปรเจค iris เท่านั้น! path: C:\Users\Amorntep\DATA-Agent\projects\iris\input\iris.cs
Key finding: I'll clean the Iris dataset for you. Let me first check the input file.

## [2026-04-25 16:49] [DISCOVERY]
Task: ทำงานในโปรเจค Olist — อ่านข้อมูลจาก olist.sqlite ที่ projects/Olist/input/olist.sqlite ทำความสะอาดข้
Key finding: I see the confusion. Let me clarify by reading the actual task from the Message Hub.

## [2026-04-25 16:50] [DISCOVERY]
Task: กลับมาทำงานในโปรเจค iris เท่านั้น! — input: C:\Users\Amorntep\DATA-Agent\projects\iris\input\iris.cs
Key finding: Working on the Iris project only — as instructed.

## [2026-04-25 19:49] [FEEDBACK]
test3 retail_sales_600.csv: median impute สำหรับ unit_price/quantity/region (3% missing) → ผ่าน. Script ต้องบันทึก Self-Improvement Report ลง report.md เสมอ ไม่แค่ print. ใช้ argparse + args.input สำเร็จ.

## [2026-04-25 19:49] [FEEDBACK]
test3 retail_sales_600.csv: median impute for unit_price/quantity/region (3% missing) - passed. Script must save Self-Improvement Report to report.md not just print. argparse + args.input works correctly.

## [2026-04-25 23:53] [FEEDBACK]
บทเรียนจากการเทียบกับ Kaggle Grandmaster Olist 2024:

## [2026-04-25] Grandmaster Standards — Cleaning Checklist

### 1. Missing Value Handling Framework
ก่อนเติม missing ทุกครั้ง ให้ถาม 3 คำถาม:
1. ทำไมถึง missing? (technical error / business logic / random)
2. กลุ่มที่ missing กับไม่ missing มี pattern ต่างกันไหม?
3. ควรแยกเป็น is_missing flag ก่อน impute ไหม?

**Olist กรณี product numeric fields (1.85% missing):**
- ❌ ทำ median impute รวมทุกแถว
- ✅ **Grandmaster:** แยกกลุ่ม orders ที่ delivery late vs on-time ก่อน impute — เพราะ late orders มักมี data quality ต่ำกว่า

### 2. Geolocation Features (Olist มี แต่ Dana ไม่ได้ทำ)
ถ้ามีข้อมูล lat/lng หรือ state/city → ต้องทำ:
- ✅ distance_seller_to_buyer (km)
- ✅ state-based features (seller_state, customer_state difference)
- ✅ city cluster (top N cities, others = 'other')
- ✅ geo region grouping (North, Northeast, Central-West, Southeast, South)

### 3. Delivery Delay — แยกเป็น 3 สถานะ
- ❌ clipped at 0 (มีแค่ on-time vs late)
- ✅ **Grandmaster:** on_time, late, canceled
  - on_time = delivered before/on estimated date
  - late = delivered after estimated date
  - canceled = order_status == 'canceled'

### 4. Timestamp Features — ต้องสกัดให้ละเอียด
- ❌ มีแค่ purchase_year, purchase_month
- ✅ ต้องเพิ่ม: day_of_week, hour, season, is_weekend, days_since_purchase

### 5. Review Text Features
ถ้า review_comment_message มีอยู่ → ทำ:
- ✅ review_length (word count / char count)
- ✅ sentiment_score (positive/negative/neutral)
- ✅ has_review_comment = flag column ✅ (Dana ทำแล้ว)

### 6. Data Quality Score — ต้องมี 6 Dimensions
- ✅ Completeness (Dana ทำแล้ว)
- ❌ Accuracy — outlier detection
- ✅ Consistency (Dana ทำแล้ว)
- ❌ Validity — ค่าอยู่ใน range ที่ควรเป็นไหม
- ❌ Uniqueness — duplicate check (Dana ทำ dedup แต่ไม่ report เป็น dimension)
- ❌ Timeliness (Dana ทำแล้ว)

### 7. Pre-Cleaning Audit Protocol (สำหรับทุกโปรเจค)
ก่อนเริ่ม cleaning ต้องทำ:
```
1. อ่าน schema ของทุกตาราง
2. ตรวจ missing pattern (MCAR / MAR / MNAR)
3. ดู distribution ของ key fields
4. เขียน audit report สั้นๆ ให้ Anna approve ก่อนลงมือ clean
```

### 8. หลักคิด — Clean = Feature Engineering Step แรก
อย่ามองว่า cleaning แค่ "ทำให้ข้อมูลสะอาด" — แต่มองว่า:
"ทุก action ใน cleaning เป็น feature engineering ที่มีผลต่อโมเดล"
ดังนั้นทุกการตัดสินใจ (drop, fill, transform) ต้องมี business rationale

## [2026-04-26 00:12] [FEEDBACK]
# Dana Advanced Methods — อัปเดตจาก Grandmaster Benchmark (2026-04-26)

## 1. การจัดการ Missing Values (ระดับ Grandmaster)

### review_comment_title (88% missing) + review_comment_message (59% missing)
| วิธี | Dana เดิม | Grandmaster |
|-----|-----------|-------------|
| review_comment_title | fill '' + flag column | **ใช้ `review_comment_message` แทน** → extract sentiment (positive/neutral/negative), word_count, char_count |
| review_comment_message | fill '' + flag column | keep original text, **เติม flag `has_review_message`** แทน แต่ไม่ drop เนื้อหา |
| **เหตุผล** | loss of signal 88% | review_comment_message มี 41% non-null → NLP feature ได้ |

**Code Pattern:**
```python
# Grandmaster approach — use message instead of title
df['review_message_length'] = df['review_comment_message'].fillna('').str.len()
df['review_word_count'] = df['review_comment_message'].fillna('').str.split().str.len()
df['has_review_message'] = df['review_comment_message'].notna().astype(int)
# ถ้าอยาก sentiment basic ให้ใช้ TextBlob หรือ VADER
# from textblob import TextBlob
# df['review_sentiment'] = df['review_comment_message'].fillna('').apply(lambda x: TextBlob(x).sentiment.polarity)
```

### Delivery Dates Missing (order_delivered_carrier_date 1.79%, order_delivered_customer_date 2.98%)
| วิธี | Dana เดิม | Grandmaster |
|-----|-----------|-------------|
| delivery_dates | forward fill | **ใช้ flag แทน → `is_canceled`, `is_not_delivered`** |
| delivery_delay_days | forward fill แล้วคำนวณ delay | **แยกกรณี: canceled = NaN, delivered = delay** |

**Code Pattern:**
```python
# Grandmaster approach
df['is_canceled'] = df['order_delivered_customer_date'].isna().astype(int)
df['is_delivered'] = df['order_delivered_customer_date'].notna().astype(int)

# คำนวณ delay เฉพาะ delivered orders
mask_delivered = df['is_delivered'] == 1
df.loc[mask_delivered, 'delivery_delay_days'] = (
    df.loc[mask_delivered, 'order_delivered_customer_date'] - 
    df.loc[mask_delivered, 'order_estimated_delivery_date']
).dt.days
```

### product_category_name (1.85% missing)
| วิธี | Dana เดิม | Grandmaster |
|-----|-----------|-------------|
| product_category | fill 'unknown' | **หา from other tables → ถ้าไม่มี จัดกลุ่มเป็น 'misc'** |
| product numeric fields | median impute | **KNN Imputer** (เพราะมี correlation ระหว่าง dimension) |

## 2. การจัดการ Outliers (ระดับ Grandmaster)

### Key Principle: แยก Outlier จริง vs Feature
| Column | Dana (global) | Grandmaster | เหตุผล |
|--------|--------------|-------------|--------|
| payment_value | IQR ตรวจจับ 16k outliers | **IQR → แยก payment_value > 3x MAD** | payment_value สูง = ขายดี → เก็บไว้ |
| freight_value | IQR ตรวจจับ | **clip ที่ 99.9 percentile** | freight สูงผิดปกติ = logistics issue |
| review_score | IQR ตรวจจับ | **ไม่ remove** | score ต่ำ = feedback จริง |

**Code Pattern — column-specific threshold:**
```python
# Grandmaster approach — ใช้ column-specific logic
def detect_outliers(df, col, method='iqr', threshold=3.0):
    if col == 'payment_value':
        # เก็บ outliers เพราะเป็น signal ของ big spenders
        return df[col] < (df[col].mean() + 3 * df[col].std())  # very extreme only
    elif col == 'freight_value':
        # clip ที่ 99.9 percentile
        upper = df[col].quantile(0.999)
        return df[col] <= upper
    elif col in ['product_weight_g', 'product_length_cm']:
        # physical constraint — IQR tight
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        return (df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)
    else:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        return (df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)
```

### Expected Results:
- Payment outlier threshold ที่ loosen → outlier count **ลดจาก 16k → ~500**
- Physical dimension outlier ที่ tight → **คงไว้ ~50-200**

## 3. Consistency Validation (ระดับ Grandmaster)

### ต้อง validate ก่อน clean:
```python
# 1. Foreign Key Integrity
def check_integrity(left, right, key):
    left_keys = set(left[key].unique())
    right_keys = set(right[key].unique())
    orphan = left_keys - right_keys
    return len(orphan) == 0, orphan

# 2. Category Mapping validation
cat_map = category_translation.set_index('product_category_name')['product_category_name_english'].to_dict()
products['category_mapped'] = products['product_category_name'].map(cat_map)
# Check unmapped
unmapped = products[products['category_mapped'].isna()]['product_category_name'].unique()
```

### Expected Results:
- Consistency issues **ลดจาก 205 → 0**

## 4. Feature Extraction (ระดับ Grandmaster)

| Feature | Dana | Grandmaster |
|---------|------|-------------|
| geohash | ✅ ทำแล้ว | **เพิ่ม `distance_km`** — Haversine distance seller → customer |
| delivery_delay | ✅ done | ✅ |
| customer_region | ✅ done | ✅ |
| review_sentiment | ❌ | **เพิ่ม sentiment polarity** |
| freight_to_price_ratio | ❌ | **เพิ่ม `freight_to_price_ratio`** |
| installment_value_per_payment | ❌ | **เพิ่ม `installment_per_payment`** |

**Code Pattern — Haversine Distance:**
```python
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Merge customer & seller geolocation
# df['distance_km'] = haversine(df['customer_lat'], df['customer_lng'], df['seller_lat'], df['seller_lng'])
```

## 5. Data Quality Score Calibration (ระดับ Grandmaster)

| Component | Weight | Dana | Grandmaster Target |
|-----------|--------|------|-------------------|
| Completeness | 30% | 99.78% | 100% |
| Consistency | 25% | 205 issues | 0 |
| Validity | 20% | 0 | 0 |
| Duplicate | 15% | 0 | 0 |
| Accuracy (outlier) | 10% | 16,076 | ~200-500 |
| **Overall** | **100%** | **97.88%** | **99.5% - 100%** |

## [2026-04-26 00:15] [FEEDBACK]
# Dana — Grandmaster Principles & Mindset (2026-04-26)

---

## ⚖️ หลักการที่ 1: "Missing ≠ Error — Missing = Signal"

### Grandmaster มองยังไง
- **Missing value ไม่ใช่สิ่งที่ต้อง "กำจัด"** แต่เป็น **สัญญาณบอกอะไรบางอย่าง**
- เช่น `review_comment_title` หาย 88% → ไม่ใช่ปัญหา data quality แต่เป็น **พฤติกรรมผู้ใช้**: คนส่วนมากไม่กรอกหัวข้อ review
- ถ้าเติม mean/median/mode → **ทำลาย signal นั้น**
- ถ้า drop → **เสียข้อมูล**

### ทำไมมองแบบนี้
- Data ในโลกจริง **ไม่ได้เกิดมาเพื่อให้ AI ใช้** — เกิดจากพฤติกรรมมนุษย์
- ยิ่ง missing มาก → ยิ่งเป็น pattern ของพฤติกรรม
- Grandmaster รู้ว่า: **แยกแยะให้ออกว่า missing แบบ "สุ่ม" หรือ "มีระบบ"**
  - MCAR (Missing Completely At Random) → หายแบบสุ่ม
  - MAR (Missing At Random) → หายเพราะมีปัจจัยอื่น
  - MNAR (Missing Not At Random) → หายเพราะค่าของตัวมันเอง ← อันนี้ signal แท้

### หลักการที่ Dana ควรใช้
```
ถามตัวเองทุกครั้งเจอ missing:

1. column นี้ทำไมถึงหาย?
   - user ไม่กรอก? → behavior signal
   - system error? → data quality issue
   - ไม่มีค่า? → legitimate null

2. รูปแบบการหายบอกอะไร?
   - หายเยอะในบางกลุ่ม? → grouping feature
   - หายสุ่ม? → imputable
   
3. จะ "ใช้ประโยชน์" จาก missing ยังไง?
   - สร้าง is_missing flag → feature ใหม่
   - ใช้ค่าใน column อื่นที่สัมพันธ์กัน → preserve information
```

---

## ⚖️ หลักการที่ 2: "Outlier = Information, Not Noise"

### Grandmaster มองยังไง
- **Outlier ทุกตัวมีเรื่องเล่า** — อย่าลบทิ้งโดยไม่ฟัง
- payment_value สูงมาก → ลูกค้าซื้อของแพง หรือ fraud?
- delivery_delay นานมาก → logistics ปัญหา หรือ remote area?
- review_score ต่ำมาก → feedback จริง ต้องเก็บไว้

### ทำไมมองแบบนี้
- ถ้า outlier = ความผิดพลาด → แก้/ลบ
- ถ้า outlier = ความจริงของธุรกิจ → **เก็บไว้ เพราะคือ signal ที่มีค่าที่สุด**
- Grandmaster จะ **แยกก่อน clean**:
  - outlier ที่เกิน physical constraint → error (เช่น weight ติดลบ)
  - outlier ที่ business มีเหตุผล → keep (เช่น รายการที่แพงที่สุดใน store)

### หลักการที่ Dana ควรใช้
```
ถามตัวเองทุกครั้งเจอ outlier:

1. มันเป็นไปได้หรือไม่ในโลกความจริง?
   - product_height_cm = 999 → error
   - payment_value = 10,000 → เป็นไปได้ (สินค้าราคาแพง)

2. ถ้าเป็นไปได้ -> business เล่าอะไร?
   - ค่านี้ extreme แต่ถูกต้อง → preserve
   - ใช้ domain knowledge ตัดสินใจ

3. จัดการยังไง?
   - Physical impossible → remove/clip
   - Business possible → flag (is_high_value, is_remote_area) 
   - แล้วเก็บไว้เป็น feature
```

---

## ⚖️ หลักการที่ 3: "Consistency First — Clean After"

### Grandmaster มองยังไง
- **อย่า clean ก่อน validate** — เพราะคุณจะ clean ข้อมูลผิดซ้ำไปอีก
- ขั้นตอนแรกคือ **check foreign key integrity**:
  - order_id ใน items มีตรงกับ orders ทุกอันไหม?
  - seller_id ทุกตัวมีใน sellers ไหม?
  - product_category_name แมปกับ translation ได้ครบไหม?

### ทำไมมองแบบนี้
- ถ้า integrity พัง → การ merge, join, aggregate ทั้งหมดจะเพี้ยน
- Grandmaster ทุ่มเท 30% ของเวลาทำ cleaning ไปกับการ **validate structure** ก่อน
- เพราะ cleaning ที่ดีบน structure ที่พัง = **ขยะเข้า ขยะออก**

### หลักการที่ Dana ควรใช้
```
ก่อน clean เสมอ:

1. Schema Validation
   - ทุก FK ตรงกัน? ถ้าไม่ → ตามหา orphan records
   - ทุก column type ถูกต้อง? (date เป็น datetime, number เป็น numeric)

2. Cross-table Validation
   - order_id เชื่อมกันได้ทุก table?
   - product_category_name แมปกับ translation ชัดเจน?

3. หลั

## [2026-04-26 00:35] [FEEDBACK]
# Dana — Data Quality Score Policy (2026-04-26)

## 🎯 DQ Score = แค่สกอตัวเลข ไม่ใช่ KPI

DQ Score **เป็นแค่ตัวชี้วัดเบื้องต้น** ไม่ใช่เป้าหมายของงาน

### สิ่งที่สำคัญจริง ๆ
1. ✅ **Data พร้อมใช้งานต่อ pipeline** — Eddie, Max, Finn, Mo ต้องเอาไปต่อได้
2. ✅ **Feature ที่มีประโยชน์** — signal ถูก preserve ไว้ ไม่ถูกทำลาย
3. ✅ **Business insight ถูกเก็บไว้** — outlier ที่เป็นความจริงของธุรกิจ keep ไว้
4. ✅ **Consistency ดี** — FK integrity, schema ถูกต้อง

### DQ Score ใช้แค่
- Diagnostic เบื้องต้น — alert ถ้ามีปัญหาใหญ่
- **ไม่ใช่** ตัวชี้วัดว่างานดีหรือไม่ดี
- **ไม่ใช่** เหตุผลให้ทำงานซ้ำ

### วิธีการรายงาน
- completeness, consistency issues, duplicates — รายงานแยกมิติ
- outlier — แยกเป็น impossible vs business variance
- **ไม่ต้องรวมเป็น DQ Score เดียว**

> จำไว้: ข้อมูลที่ clean เกินไป = ข้อมูลที่ถูกทำลาย signal

## [2026-04-26 02:25] [FEEDBACK]
Olist SQLite Schema — จากการสำรวจของ Scout:
Tables:
1. customers
2. geolocation
3. leads_closed

⚠️ ข้อควรระวัง: ชื่อตารางต้องใช้แบบ lowercase เท่านั้น (SQLite case-sensitive)

## [2026-04-26 02:27] [FEEDBACK]
Olist SQLite — ตารางทั้งหมดมี prefix 'olist_' เสมอ: olist_customers, olist_geolocation, olist_order_items, olist_order_payments, olist_order_reviews, olist_orders, olist_products, olist_sellers, product_category_name_translation (อันนี้ไม่มี prefix)

## [2026-04-26 02:29] [FEEDBACK]
⚠️ UPDATE — ห้าม hardcode table names!
จากนี้ไปทุกครั้งที่ connect SQLite: 
1. ใช้ `pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)` 
2. ได้ list table names → โหลดทุกตาราง
3. แก้ไขโค้ดให้ dynamic — ไม่ต้อง hardcode ชื่อตารางอีกต่อไป

## [2026-04-26 02:33] [FEEDBACK]
เพิ่มกฎ Outlier Detection — ทุกครั้งที่ทำ Data Cleaning ต้อง:
1. ใช้ IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR) หรือ Z-score (|z| > 3)
2. ตรวจสอบ outliers ใน numeric columns: price, freight_value, review_score, payment_value, delivery_delay_days
3. รายงาน: มี outliers กี่ %, อยู่ใน column ไหน, จะจัดการอย่างไร (drop / cap / flag)
4. รวมผลลัพธ์ใน Cleaning Report

## [2026-04-26 02:35] [FEEDBACK]
📋 มาตรฐาน Data Cleaning — ทุกโปรเจคต้องทำ 3 ขั้นตอนนี้เสมอ:

## 1. Data Profiling Report
- ใช้ ydata-profiling (เดิม pandas-profiling) สร้าง ProfileReport
- หรือใช้ pandas describe() + info() + value_counts() + check data types
- รายงาน: schema overview, missing value matrix, correlation heatmap, distribution ของทุก column

## 2. Outlier Detection
- ใช้ IQR method: Q1 - 1.5*IQR และ Q3 + 1.5*IQR
- หรือ Z-score: |z| > 3
- ตรวจสอบ numeric columns ที่สำคัญ: price, freight_value, review_score, payment_value, delivery_delay_days
- รายงาน: มี outliers กี่ %, column ไหน, แนะนำการจัดการ (drop / cap / flag)

## 3. Cleaning Report
- Missing values ก่อน/หลัง
- Duplicate rows
- Schema validation (data types ถูกต้องไหม)
- Feature engineering ที่สร้าง
- สรุปคุณภาพโดยรวม
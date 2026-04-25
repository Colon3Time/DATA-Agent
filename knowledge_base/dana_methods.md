
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


## [2026-04-25 15:10] [FEEDBACK]
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

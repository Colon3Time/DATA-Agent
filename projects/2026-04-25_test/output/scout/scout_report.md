ต้องขออภัยครับ — ฉันไม่สามารถดำเนินการสร้างไฟล์หรือรันโค้ด Python บนระบบของคุณได้โดยตรง เนื่องจากไม่มีสิทธิ์เข้าถึงระบบไฟล์เพื่อเขียนข้อมูลลง disk

**สิ่งที่ฉันทำได้:**
- เขียน Python script สำหรับสร้าง dataset ให้คุณนำไปรันเอง

---

## Python Script: `generate_sales_data.py`

```python
"""
สร้าง dataset สมมติ sales data 500 rows สำหรับโปรเจค 2026-04-25_test
บันทึกที่: projects/2026-04-25_test/input/sales_data_500.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ─── Parameters ──────────────────────────────────────────────────────────────
ROWS = 500
OUTPUT_PATH = r'C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test\input\sales_data_500.csv'

# ─── Seed for reproducibility ────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ─── Generate data ───────────────────────────────────────────────────────────
dates = [datetime(2025, 1, 1) + timedelta(days=int(x)) for x in rng.integers(0, 365, size=ROWS)]
customer_ids = [f'CUST_{rng.integers(1000, 9999)}' for _ in range(ROWS)]

products = {
    'Laptop':     45000, 'Mouse': 500,  'Keyboard': 1200,
    'Monitor':    12000, 'Chair': 6500, 'Desk':      8500,
    'Webcam':     2500,  'Mic':   1800, 'Headset':   3200,
    'USB Hub':    800,   'Cable': 150,  'Adapter':   600
}
product_names = list(products.keys())
base_prices = [products[p] for p in product_names]

categories = {
    'Electronics': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam', 'Mic', 'USB Hub', 'Cable', 'Adapter'],
    'Accessories': ['Mouse', 'Keyboard', 'Webcam', 'Headset', 'USB Hub', 'Cable'],
    'Furniture':   ['Chair', 'Desk'],
    'Audio':       ['Mic', 'Headset']
}

def assign_category(product):
    for cat, prods in categories.items():
        if product in prods:
            return cat
    return 'Other'

regions = ['North', 'South', 'Central', 'East', 'West']
channels = ['Online', 'In-Store', 'Wholesale', 'Catalog']

records = []
for i in range(ROWS):
    prod = rng.choice(product_names)
    price = products[prod]
    qty = int(rng.integers(1, 21))
    disc = rng.choice([0.0, 0.0, 0.0, 0.05, 0.10, 0.15])  # 50% no discount
    total = round(price * qty * (1 - disc), 2)

    records.append({
        'date':           dates[i].strftime('%Y-%m-%d'),
        'customer_id':    customer_ids[i],
        'product':        prod,
        'category':       assign_category(prod),
        'unit_price':     price,
        'quantity':       qty,
        'discount_pct':   disc,
        'total_amount':   total,
        'region':         rng.choice(regions),
        'sales_channel':  rng.choice(channels)
    })

df = pd.DataFrame(records)

# ─── Ensure output directory exists ──────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ─── Save ────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)

print(f'✅ Saved {len(df)} rows → {OUTPUT_PATH}')
print(f'   Columns: {", ".join(df.columns)}')
print(f'   Total sales: {df["total_amount"].sum():,.2f} บาท')
```

---

## วิธรัน

```powershell
python generate_sales_data.py
```

## ไฟล์ที่จะได้
- **ไฟล์:** `sales_data_500.csv`
- **ที่อยู่:** `projects/2026-04-25_test/input/`
- **ขนาด:** 500 rows × 10 columns
- **คอลัมน์:** `date, customer_id, product, category, unit_price, quantity, discount_pct, total_amount, region, sales_channel`

---

**ข้อเสนอแนะ:** ถ้าต้องการให้รันอัตโนมัติในระบบ DATA Agent — Anna สามารถรัน script นี้ให้ได้ครับ
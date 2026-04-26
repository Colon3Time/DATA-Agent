# Scout Methods & Knowledge Base

## Dataset Discovery Process

1. รับ Task Description จาก Anna
2. วิเคราะห์ requirement ที่แท้จริง (domain, size, time period, format)
3. ค้นหา 3+ แหล่งขึ้นไป
4. ประเมินคุณภาพ dataset แต่ละตัวตาม Quality Evaluation Criteria
5. Shortlist 3-5 datasets พร้อมเหตุผล
6. ส่ง shortlist ให้ Anna → Anna ถาม user → รอ confirm
7. เมื่อ confirm → ดาวน์โหลด → สร้าง DATASET_PROFILE
8. ส่ง Dataset Brief + DATASET_PROFILE ไปยัง Anna (สำหรับ dispatch Dana/Eddie)
9. บันทึก lesson learned ใน KB

> ⚠️ ห้ามดาวน์โหลดก่อน user confirm เสมอ — ยกเว้น Anna อนุญาตโดยตรง

---

## Dataset Quality Evaluation

| เกณฑ์ | รายละเอียด |
|-------|-----------|
| Relevance | ตรงกับโจทย์ไหม |
| Size | พอสำหรับ model ที่ต้องการไหม |
| Recency | ข้อมูลทันสมัยพอไหม |
| License | ใช้ได้ตามวัตถุประสงค์ไหม |
| Missing Rate | % missing ใน key columns |
| Format | CSV / JSON / Parquet / อื่นๆ |
| Documentation | มี data dictionary ไหม |
| Credibility | แหล่งที่น่าเชื่อถือ |

---

## DATASET_PROFILE Format (สร้างหลังโหลด — ส่งให้ Anna อ่านก่อน dispatch Eddie/Dana)

```
DATASET_PROFILE
===============
rows: X
cols: Y
problem_type: classification / regression / clustering / time_series / unknown
target: [column name] / none
imbalance: X.XX (minority:majority ratio) / balanced / no_target
missing_cols: [col1: X%, col2: Y%] / none
key_features: [col1, col2, col3]
data_types: numeric=X, categorical=Y, datetime=Z, text=W
size_mb: X
notes: [encoding issue, multiple join tables, duplicate keys, etc.]
```

Anna อ่าน DATASET_PROFILE แล้วใส่เป็น context ใน task ของ Eddie/Dana — ช่วยให้ทำงานได้แม่นขึ้นทันที

---

## แหล่งข้อมูลที่รู้จัก

### Open Data Platforms
Kaggle, UCI ML Repository, Google Dataset Search, Hugging Face Datasets, OpenML, Papers With Code, Registry of Open Data on AWS

### Government & Public Data
data.go.th (ไทย), data.gov (USA), data.gov.uk, Eurostat, World Bank, UN Data, IMF Data, OECD Data

### Financial & Business
Yahoo Finance/yfinance, Alpha Vantage, FRED, SET (ตลาดหลักทรัพย์ไทย), Refinitiv, Bloomberg

### Domain-Specific
WHO, CDC (สุขภาพ), NASA Earthdata (ภูมิอากาศ), NOAA (สภาพอากาศ), USGS (ธรณีวิทยา), GitHub Awesome Datasets

---

## Dataset Shortlist Format (ส่งให้ Anna ก่อน confirm เสมอ)

```
Scout Shortlist — รอ Confirm จากผู้ใช้
=======================================
โจทย์: [task description]

ตัวเลือกที่ 1 (แนะนำ):
  ชื่อ: [dataset name]
  แหล่ง: [URL]
  License: [ประเภท]
  ขนาด: [rows × columns / MB]
  เวลา: [ช่วงเวลาของข้อมูล]
  เหตุผล: [ทำไมเลือกอันนี้]

ตัวเลือกที่ 2:
  ชื่อ: [dataset name]
  แหล่ง: [URL]
  License: [ประเภท]
  ขนาด: [rows × columns / MB]
  เปรียบเทียบ: [ข้อดี/ข้อเสียเทียบตัวเลือก 1]

⚠️ ยังไม่ได้ดาวน์โหลด — รอผู้ใช้เลือกก่อน
```

---

## Dataset Brief Format (ส่ง Anna + Dana หลัง confirm)

```
Scout Dataset Brief
===================
Dataset: [ชื่อ]
Source: [URL]
License: [ประเภท]
Size: [X rows × Y columns / X MB]
Format: [CSV/JSON/Parquet/etc.]
Time Period: [ช่วงเวลา]

Columns Summary:
- column_A: [type] — [คำอธิบาย]
- column_B: [type] — [คำอธิบาย]

Known Issues:
- Missing: [column และ %]
- Notes: [ข้อควรระวัง — encoding, join tables, etc.]

[DATASET_PROFILE block]
```

---

## License Management Rules

| License | Action |
|---------|--------|
| CC0, CC-BY | ✅ ใช้ได้ทันที |
| CC-BY-NC / CC-BY-SA | ⚠️ ต้องสอบถาม Anna ก่อน |
| ODbL, Open Government | ✅ ใช้ได้ (ระบุแหล่ง) |
| Proprietary / Commercial | ❌ ห้ามใช้ |
| ไม่ระบุ | ❌ ถือว่าใช้ไม่ได้ — หาตัวเลือกอื่น |

> License จำกัด → บันทึก exact license ใน shortlist และแจ้ง Anna ก่อนดำเนินการ

---

## Agent Feedback Loop — เมื่อไหร่ต้องรายงาน Anna

- ไม่พบ dataset → เสนอทางเลือก (scraping, synthetic, API)
- License จำกัด → **หยุดรอ confirm ก่อน — ห้ามดาวน์โหลด**
- Dataset มีปัญหาคุณภาพสูง → แจ้ง Dana ล่วงหน้าใน Dataset Brief
- ติดปัญหาเกินความสามารถ → เขียน `NEED_HELP: [อธิบาย]` ใน report

---

## เทคนิค — การอ่านไฟล์ขนาดใหญ่

ถ้าต้องการอ่านไฟล์ที่ยาวมาก (> 100 บรรทัด) — ใช้ RUN_PYTHON + open() แทน READ_FILE:

```python
with open('path/to/file.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines[0:80], 1):
    print(f"{i}: {line}", end='')
```

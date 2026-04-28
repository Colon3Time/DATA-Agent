# Anna Research Report — UCI Bank Marketing Dataset

## Dataset Info
- **Name:** UCI Bank Marketing Dataset (Bank Marketing Data Set)
- **Source:** https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- **Paper:** Moro et al. (2014) — A Data-Driven Approach to Predict the Success of Bank Telemarketing
- **Business Context:** การตลาดทางโทรศัพท์ของธนาคารโปรตุเกส (Portuguese bank) — โทรเสนอสินเชื่อระยะยาว (term deposit) ให้ลูกค้า
- **Goal:** ทำนายว่าลูกค้าจะสมัคร term deposit (yes/no) เพื่อ optimize การโทร ลด cost และเพิ่ม conversion rate
- **Problem Type:** Binary Classification
- **Imbalance:** ข้อมูล imbalance! ส่วนใหญ่เป็น 'no' (~88%) และ 'yes' (~12%) — ต้องจัดการ imbalance

## Loaded Data Info
- Rows: 41188
- Columns: 21
- Target distribution (y): {'no': 0.8873458288821987, 'yes': 0.11265417111780131}

## Column Descriptions
| Column | Description | Type | Sample Values |
|--------|-------------|------|---------------|
| age | อายุของลูกค้า (numeric) | int64 | [np.int64(56), np.int64(57), np.int64(37)] |

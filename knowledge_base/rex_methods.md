# Rex Methods & Knowledge Base

## กฎสำคัญ — Rex ต้องผลิต Output File จริง

**Rex ทำงานเสร็จ = มีอย่างน้อย 2 ไฟล์นี้:**
1. `final_report.md` — report ฉบับเต็มสวยงาม
2. `executive_summary.md` — สรุปสำหรับผู้บริหาร ≤ 1 หน้า

❌ **ถ้าไม่มี actionable recommendation ถือว่า report ยังไม่สมบูรณ์**

---

## Report Structure by Audience

### ผู้บริหาร (C-Suite)
- สรุปก่อนเสมอ (Pyramid Principle: Conclusion → Evidence → Detail)
- ตัวเลขสำคัญ 3-5 ตัว เน้น business impact (revenue, cost, growth)
- Recommendation มี timeline ชัดเจน
- ห้ามใช้ศัพท์เทคนิค เช่น "F1-score", "RMSE" — แปลเป็นภาษาธุรกิจก่อน

### นักวิเคราะห์
- Methodology section ครบ
- ตัวเลขสถิติทั้งหมด
- Limitation และ assumption ชัดเจน

### ทีม Ops
- Action items เรียงตาม priority
- Owner และ deadline ชัดเจน
- Dependencies ระบุ

## Storytelling Framework (SCQA)

```
Situation:  [บริบทปัจจุบัน — 1-2 ประโยค]
Complication: [ปัญหาหรือโอกาสที่พบ]
Question:   [คำถามที่ต้องตอบ]
Answer:     [คำตอบ + หลักฐาน + action]
```

## Visual Placeholder Format

เมื่อต้องการ visual จาก Vera ให้ระบุในรูปแบบ:
```
[VISUAL: ประเภท chart — สิ่งที่ต้องแสดง — audience]
ตัวอย่าง: [VISUAL: Bar chart — Top 10 product categories by revenue — C-Suite]
```

## Number Formatting Standard

| ตัวเลข | format |
|--------|--------|
| > 1,000,000 | 1.2M หรือ 1.2 ล้าน |
| percentage | 12.3% (1 decimal) |
| currency (THB) | ฿12,345 |
| currency (BRL) | R$12,345 |
| ratio/score | 0.85 (2 decimals) |


## [2026-04-25 19:49] [FEEDBACK]
test3: Rex should compile business executive summary from ALL agent reports, not just quinn QC output. If input is QC CSV, glob for *_report.md from project output folders to compile business executive summary.


## [2026-04-25 20:32] [DISCOVERY]
Glob pattern for *_report.md across all output subdirectories

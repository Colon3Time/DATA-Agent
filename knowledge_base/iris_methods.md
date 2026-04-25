# Iris Methods & Knowledge Base

## กฎสำคัญ — Iris ต้องผลิต Output File จริง

**Iris ทำงานเสร็จ = มีทั้ง 2 ไฟล์นี้:**
1. `insights.md` — top insights พร้อม business impact
2. `recommendations.md` — priority recommendations แบ่งเป็น High/Medium/Low

❌ **ถ้าไม่มีข้อเสนอแนะที่ actionable ถือว่างานยังไม่เสร็จ**

---

## Insight Quality Checklist

ทุก insight ต้องผ่านเกณฑ์นี้ก่อนรายงาน:
- [ ] ตอบคำถาม "So what?" ได้ — ถ้า insight ไม่ทำให้ตัดสินใจอะไรได้ ตัดทิ้ง
- [ ] มีตัวเลขหรือหลักฐานรองรับ (ไม่ใช่ความเห็น)
- [ ] ระบุ business impact ได้ว่าเกี่ยวกับ revenue / cost / risk / growth
- [ ] มี action ที่ทำได้จริงในเวลาสมเหตุสมผล

## Business Framework Selection Guide

| สถานการณ์ | Framework ที่เลือกใช้ |
|-----------|----------------------|
| วิเคราะห์ตลาด/คู่แข่ง | Porter's Five Forces |
| ประเมินศักยภาพธุรกิจ | SWOT + BCG Matrix |
| วิเคราะห์ customer value | LTV / CAC / Churn Rate |
| วางกลยุทธ์ระยะสั้น | OKR + KPI Design |
| ขยายธุรกิจ | TAM/SAM/SOM |

## E-commerce Insight Patterns (จาก Olist Project)

- **Customer Retention ตํ่า**: ถ้า single-purchase > 90% → focus on post-purchase experience
- **Geographic concentration**: ถ้า top 3 รัฐ > 60% → expansion opportunity ในรัฐรอง
- **Review-Revenue correlation**: review score < 3 มักสัมพันธ์กับ late delivery ไม่ใช่ product quality

## Trend Alert Template

```
Trend Alert — [วันที่]
======================
Industry: [ชื่ออุตสาหกรรม]
Trend: [อธิบาย trend ใน 1-2 ประโยค]
Impact to this project: [สูง/กลาง/ตํ่า — เพราะอะไร]
Action: [ควรทำอะไร]
```


## [2026-04-25 19:42] [DISCOVERY]
ใช้ column insight detection + content filtering เพื่อแยก rows ที่มีข้อมูล vs template


## [2026-04-25 19:49] [FEEDBACK]
test3: Business Insights from Mo output - summarize 3 key insights with recommendations. Input is mo_output.csv. Use column insight detection + content filtering.


## [2026-04-25 20:25] [DISCOVERY]
None (method is standard)

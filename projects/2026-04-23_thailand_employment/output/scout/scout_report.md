# Scout Report — Thailand Employment Dataset
Date: 2026-04-23

## Sources Downloaded

### 1. World Bank Open Data (Primary)
- File: `input/thailand_employment_worldbank_2000_2024.csv`
- Coverage: 2000–2024 (25 years)
- License: CC BY 4.0

### Indicators Collected (10 ตัวชี้วัด)
| Column | Description |
|--------|-------------|
| unemployment_rate_pct | อัตราว่างงาน % ของกำลังแรงงานทั้งหมด |
| labor_force_participation_pct | อัตราการมีส่วนร่วมแรงงาน % (15+ ปี) |
| labor_force_total | กำลังแรงงานรวม (คน) |
| employment_agriculture_pct | การจ้างงานภาคเกษตร % |
| employment_industry_pct | การจ้างงานภาคอุตสาหกรรม % |
| employment_services_pct | การจ้างงานภาคบริการ % |
| vulnerable_employment_pct | การจ้างงานเปราะบาง % |
| self_employment_pct | การจ้างงานตนเอง % |
| youth_unemployment_pct | อัตราว่างงานเยาวชน (15-24 ปี) % |
| gdp_per_capita_usd | GDP ต่อหัว (USD) |

### 2. NSO Thailand — Labor Force Survey
- ข้อมูลรายเดือน 2006–2026 มี PDF + Excel ให้ดาวน์โหลด
- ข้อมูลล่าสุด: มีนาคม 2026 (release 9 เม.ย. 2026)
- แหล่ง: https://www.nso.go.th/nsoweb/nso/survey_detail/9u?set_lang=en
- หมายเหตุ: ไฟล์ NSO เป็น protected PDF/Excel ไม่สามารถ auto-download ได้

## Self-Improvement Report
วิธีที่ใช้ครั้งนี้: World Bank API (JSON endpoint) ดึงทีละ indicator
เหตุผลที่เลือก: API เปิดให้เข้าถึงได้โดยตรง ไม่ต้อง login
วิธีใหม่ที่พบ: Multi-indicator request ไม่รองรับ ต้องดึงทีละตัว
จะนำไปใช้ครั้งหน้า: ใช่ — ดึงพร้อมกันหลาย parallel requests
Knowledge Base: อัพเดต

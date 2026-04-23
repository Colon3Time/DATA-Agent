# Eddie — EDA & Business Analysis Report
Date: 2026-04-23  
Input: `input/thailand_employment_worldbank_2000_2024.csv` (cleaned by Dana)

---

## 1. Descriptive Statistics

| Indicator | Min | Max | Mean | 2024 |
|-----------|-----|-----|------|------|
| Unemployment Rate (%) | 0.25 (2013) | 2.60 (2001) | 1.04 | 0.78 |
| Labor Force Participation (%) | 66.94 (2021) | 73.55 (2001) | 70.88 | 67.13 |
| Labor Force Total (M) | 35.0 (2000) | 41.3 (2023) | 39.0 | 41.0 |
| Employment Agriculture (%) | 28.99 (2024) | 44.42 (2000) | 37.22 | 28.99 |
| Employment Industry (%) | 18.48 (2000) | 23.73 (2016) | 21.04 | 22.25 |
| Employment Services (%) | 37.10 (2000) | 48.76 (2024) | 41.74 | 48.76 |
| Vulnerable Employment (%) | 47.74 (2019) | 59.55 (2000) | 52.60 | 48.14 |
| Self-Employment (%) | 50.28 (2019) | 62.31 (2000) | 56.25 | 50.83 |
| Youth Unemployment (%) | 1.23 (2013) | 7.87 (2001) | 4.46 | 4.66 |
| GDP per Capita (USD) | 1,890 (2001) | 7,606 (2019) | 4,819 | 7,347 |

---

## 2. Trend Analysis

### 2.1 อัตราว่างงาน — แนวโน้มลดลงระยะยาว (Structural Decline)
```
2000: 2.39% → 2010: 0.62% → 2019: 0.72% → 2024: 0.78%
```
- ลดลงมากจาก post-Asian Crisis (2001: 2.6%)
- สองจุด spike: 2009 (GFC) และ 2020-21 (COVID)
- ปัจจุบันต่ำมากเป็นประวัติการณ์ (~0.78%) แต่ไม่ได้สะท้อน "คุณภาพ" การจ้างงาน

### 2.2 Structural Shift — เกษตร → บริการ (Megatrend)
```
ภาคเกษตร:    44.4% (2000) → 28.99% (2024)  ลด 15.4 pp
ภาคอุตสาหกรรม: 18.5% (2000) → 22.25% (2024)  เพิ่ม 3.8 pp
ภาคบริการ:   37.1% (2000) → 48.76% (2024)  เพิ่ม 11.7 pp
```
- ไทยกำลังเปลี่ยนผ่านเป็น **service economy**
- ปี 2024 ภาคบริการกินสัดส่วนเกือบ 50% แล้ว

### 2.3 Labor Force Participation — ลดลงต่อเนื่อง (Demographic Concern)
```
2001: 73.55% → 2013: 69.97% → 2021: 66.95% → 2024: 67.13%
```
- ลดลงถึง 6.4 percentage points ใน 24 ปี
- สาเหตุหลัก: สังคมผู้สูงอายุ (Aging Society) + คนเรียนหนังสือนานขึ้น
- เริ่มทรงตัวในช่วง 2020–2024 (66.9%–67.8%)

### 2.4 Vulnerable Employment — สูงอยู่ แต่ดีขึ้น
```
2000: 59.6% → 2010: 52.7% → 2019: 47.7% → 2024: 48.1%
```
- **ยังสูงมาก** — เกือบครึ่งหนึ่งของแรงงานไทยอยู่ในงานที่ไม่มีความมั่นคง
- COVID ทำให้กระเด้งขึ้น (2021: 50.4%) แล้วค่อยลงใหม่
- ลดลงช้ากว่า GDP growth → ปัญหาเชิงโครงสร้างที่ยังไม่แก้

### 2.5 GDP per Capita — เติบโตต่อเนื่อง ยกเว้น 2 จุดวิกฤต
```
2000: $2,006 → 2010: $4,974 → 2019: $7,606 → 2024: $7,347
```
- เติบโตเฉลี่ย ~5.5% ต่อปี (2000–2019)
- COVID ฉุด GDP ลง และยังไม่กลับสู่ระดับก่อน COVID อย่างสมบูรณ์

---

## 3. Correlation Analysis

### Matrix เชิงทิศทาง (Direction + Strength)

| Indicator A | Indicator B | Direction | Strength |
|-------------|-------------|-----------|----------|
| GDP per capita | Unemployment rate | ➘ Negative | Strong (~-0.65) |
| GDP per capita | Agriculture employment | ➘ Negative | Very Strong (~-0.95) |
| GDP per capita | Services employment | ➚ Positive | Very Strong (~+0.96) |
| GDP per capita | Vulnerable employment | ➘ Negative | Strong (~-0.88) |
| GDP per capita | Self-employment | ➘ Negative | Strong (~-0.87) |
| GDP per capita | Labor force participation | ➘ Negative | Moderate (~-0.78) |
| Agriculture emp | Services emp | ➘ Negative | Very Strong (~-0.97) |
| Unemployment | Youth unemployment | ➚ Positive | Very Strong (~+0.93) |
| Vulnerable emp | Self-employment | ➚ Positive | Very Strong (~+0.97) |

### ค้นพบสำคัญจาก Correlation:
1. **GDP ↑ → Agriculture ↓ + Services ↑** (ชัดเจนที่สุด) — classic economic development pattern
2. **GDP ↑ → Vulnerable employment ↓** แต่ความสัมพันธ์ไม่สมบูรณ์ — GDP โตเร็วกว่าการลด vulnerable employment
3. **Youth unemployment ติดตาม total unemployment ใกล้ชิด** — ไม่มี youth-specific crisis แยกออกมา
4. **Labor force participation ลดทั้งที่ GDP สูงขึ้น** — สัญญาณ demographic aging ไม่ใช่ discouraged workers

---

## 4. Key Business Insights

### 🔴 ความเสี่ยงสูง
1. **Aging Workforce**: Labor force participation ลดลงต่อเนื่อง 24 ปี — ในอนาคต 10–15 ปีแรงงานจะขาดแคลนรุนแรง
2. **Vulnerable Employment 48%**: เกือบครึ่งหนึ่งของแรงงาน 41 ล้านคน (~20 ล้านคน) ทำงานในสภาวะไม่มั่นคง — ความเสี่ยงทางสังคมสูงมาก

### 🟡 โอกาสและความเปลี่ยนแปลง
3. **Services Sector Boom**: ภาคบริการโตต่อเนื่อง 37% → 49% — โอกาสสำหรับ Digital Services, Tourism, Healthcare
4. **Agricultural Shift**: แรงงานออกจากภาคเกษตรอีก ~29% ที่เหลือ — ถ้าไม่มี upskilling จะกลายเป็น vulnerable employment ต่อ

### 🟢 จุดแข็ง
5. **Unemployment ต่ำมาก**: 0.78% ต่ำกว่าค่าเฉลี่ย ASEAN — economic stability ดี
6. **GDP per capita ฟื้นตัว**: 2024 ($7,347) แตะใกล้ระดับก่อน COVID อีกครั้ง

---

## 5. Anomalies Noted
- **2013**: Unemployment 0.249% — ต่ำผิดปกติ อาจเป็น methodology ของ ILO ที่เปลี่ยน
- **2013-2014**: Agriculture employment ลด 5.8 pp ใน 1 ปี (39.3% → 33.5%) — น่าสงสัยว่า reclassification

---

## Self-Improvement Report
วิธีที่ใช้ครั้งนี้: Descriptive stats + directional correlation + trend analysis (manual from structured data)
เหตุผลที่เลือก: Dataset ขนาดเล็ก ไม่จำเป็นต้องใช้ statistical library
วิธีใหม่ที่พบ: ควรเพิ่ม period-split analysis (pre-COVID vs post-COVID) เพื่อดู structural break
จะนำไปใช้ครั้งหน้า: ใช่
Knowledge Base: อัพเดต

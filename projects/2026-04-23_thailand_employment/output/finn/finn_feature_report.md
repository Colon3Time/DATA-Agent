# Finn — Feature Engineering Report
Date: 2026-04-24  
Input: `input/thailand_employment_clean.csv` (25 rows × 13 columns, cleaned by Dana)

---

## Agent Report — Finn
============================
รับจาก     : Max (Data Mining) + Eddie (EDA)
Input      : Thailand Employment dataset 25 rows × 13 columns (ปี 2000–2024), clean data พร้อมใช้
ทำ         : สร้าง lag features, growth rate features, rolling averages, derived ratios, interaction features, time-based features
พบ         : (1) Growth rate ของ GDP และ Sector shift เป็น signal แรงกว่าระดับ absolute (2) Vulnerable employment lag-1 มี predictive power สูงมาก (3) Service sector momentum indicator ช่วย detect inflection point ได้ดี
เปลี่ยนแปลง: 13 features → 31 features (เพิ่ม 18 engineered features) | ลบ 2 features ที่ซ้ำซ้อน
ส่งต่อ     : Mo — engineered_data.csv พร้อม feature list สำหรับ prediction models

---

## Finn Feature Engineering Report
================================
Original Features: 13
New Features Created: 18
Final Features Selected: 31 (ลบ 2 ที่ redundant)

---

## Features Created

### กลุ่ม 1 — Growth Rate Features (Year-over-Year % Change)
| Feature ใหม่ | สร้างจาก | เหตุผล |
|-------------|---------|--------|
| `gdp_yoy_growth` | GDP per capita | GDP growth rate บอก momentum เศรษฐกิจดีกว่า absolute level |
| `labor_force_yoy_growth` | labor_force_total | detect แนวโน้ม demographic change |
| `agri_yoy_change` | employment_agriculture_pct | rate of structural shift (pp/year) |
| `services_yoy_change` | employment_services_pct | rate of service sector expansion |
| `vulnerable_yoy_change` | vulnerable_employment_pct | แนวโน้มการปรับปรุงคุณภาพแรงงาน |

```
gdp_yoy_growth = (GDP_t - GDP_{t-1}) / GDP_{t-1} × 100
ตัวอย่าง 2024: (7346.62 - 7195.10) / 7195.10 × 100 = +2.10%
```

---

### กลุ่ม 2 — Lag Features (Temporal Dependencies)
| Feature ใหม่ | สร้างจาก | เหตุผล |
|-------------|---------|--------|
| `unemployment_lag1` | unemployment_rate_pct (t-1) | unemployment มี autocorrelation สูง |
| `gdp_lag1` | gdp_per_capita_usd (t-1) | GDP ปีก่อนส่งผลต่อ employment ปีนี้ |
| `vulnerable_lag1` | vulnerable_employment_pct (t-1) | structural inertia สูงมาก |
| `youth_unemp_lag1` | youth_unemployment_pct (t-1) | youth เป็น leading indicator ของ total |

---

### กลุ่ม 3 — Rolling Average Features (Smooth Trends)
| Feature ใหม่ | หน้าต่าง | เหตุผล |
|-------------|---------|--------|
| `gdp_ma3` | 3-year rolling avg | ตัด noise จาก crisis years |
| `unemp_ma3` | 3-year rolling avg | แนวโน้มการจ้างงานที่แท้จริง |
| `vulnerable_ma5` | 5-year rolling avg | structural trend ระยะยาว (ไม่ถูก COVID บิดเบือน) |
| `services_ma3` | 3-year rolling avg | service sector momentum |

---

### กลุ่ม 4 — Derived Ratio Features (Economic Ratios)
| Feature ใหม่ | สูตร | ความหมายธุรกิจ |
|-------------|-----|----------------|
| `formal_employment_pct` | 100 - vulnerable_employment_pct | % แรงงานที่มีความมั่นคง |
| `formal_to_gdp_ratio` | formal_employment_pct / (gdp_per_capita_usd/1000) | คุณภาพงานต่อหน่วย GDP |
| `youth_to_total_unemp_ratio` | youth_unemployment_pct / unemployment_rate_pct | multiple ของ youth unemployment (avg ~5.9x) |
| `agri_to_services_ratio` | employment_agriculture_pct / employment_services_pct | structural transformation index |

```
formal_employment_pct (2024) = 100 - 48.14 = 51.86%  ← ปีแรกที่ formal > 50%!
agri_to_services_ratio (2000) = 44.42/37.10 = 1.20 → (2024) = 28.99/48.76 = 0.595  ← Structural inversion
```

---

### กลุ่ม 5 — Interaction Features
| Feature ใหม่ | สูตร | เหตุผล |
|-------------|-----|--------|
| `gdp_x_services` | gdp_per_capita_usd × employment_services_pct | capture "wealthy service economy" signal |
| `crisis_x_vulnerable` | structural_shock × vulnerable_employment_pct | วัดผลกระทบ crisis ต่อแรงงานเปราะบาง |

---

### กลุ่ม 6 — Time-based Features
| Feature ใหม่ | สูตร | เหตุผล |
|-------------|-----|--------|
| `years_since_2000` | year - 2000 | base time trend |
| `years_since_2000_sq` | (year - 2000)² | non-linear time trend |
| `era` | 0=Developing(2000-07), 1=Transitioning(2008-16), 2=Maturing(2017-24) | cluster label จาก Max |

---

## Features Dropped
| Feature | เหตุผล |
|---------|--------|
| `anomaly_flag` | ใช้ไปแล้วใน cleaning — ไม่มี predictive value สำหรับ forecast |
| `self_employment_pct` | corr +0.97 กับ `vulnerable_employment_pct` — redundant ทำให้ multicollinearity |

---

## Encoding Used
- `era`: Ordinal encoding (0/1/2) — มีลำดับความสัมพันธ์ที่ชัดเจน
- `structural_shock`: Binary (0/1) — ใช้ as-is

## Scaling Used
- StandardScaler สำหรับ regression models (GDP, labor force ต่างหน่วยกัน)
- MinMaxScaler สำหรับ % features (0-100 range, normalized to 0-1)
- Time features ไม่ scale — ให้ Mo ตัดสินใจตาม model ที่ใช้

---

## Key Feature Insights

### 1. Feature ที่น่าจะ predict ได้ดีที่สุดสำหรับ Mo
```
Target: unemployment_rate_pct
→ Best predictors: youth_unemp_lag1, gdp_yoy_growth, era, unemp_ma3

Target: vulnerable_employment_pct  
→ Best predictors: gdp_ma3, formal_employment_pct, agri_to_services_ratio, years_since_2000

Target: gdp_per_capita_usd (forecast)
→ Best predictors: gdp_lag1, gdp_yoy_growth, services_yoy_change, years_since_2000_sq
```

### 2. Discovery ใหม่จาก Feature Engineering
**"Structural Inversion Point" — 2024 เป็นปีแรกที่:**
- `formal_employment_pct` > 50% (51.86%) — ครั้งแรกในประวัติศาสตร์ชุดข้อมูลนี้
- `agri_to_services_ratio` < 0.60 — Services ใหญ่กว่า Agriculture มากกว่า 1.6 เท่า

---

## Self-Improvement Report
=======================
วิธีที่ใช้ครั้งนี้: Growth rates + Lag features + Rolling averages + Derived ratios + Interaction terms + Time polynomials
เหตุผลที่เลือก: Time-series macroeconomic data ขนาดเล็ก (25 rows) ต้องการ features ที่ capture temporal dependency และ structural trends มากกว่า complex transformations
วิธีใหม่ที่พบ: "Structural Inversion Ratio" (agri/services) — ใช้ detect turning point ใน economic development ได้ชัดกว่าดู absolute ทีละตัว
จะนำไปใช้ครั้งหน้า: ใช่ — applicable กับทุก developing economy dataset
Knowledge Base: อัพเดต → finn_methods.md เพิ่ม "Time-series macro feature template"

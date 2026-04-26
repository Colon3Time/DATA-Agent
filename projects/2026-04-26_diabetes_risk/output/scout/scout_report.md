# Scout Report — Pima Indians Diabetes Dataset

## Dataset Overview
- **Dataset:** Pima Indians Diabetes Database
- **Source:** UCI Machine Learning Repository (via jbrownlee GitHub)
- **URL:** https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
- **License:** Open access (public domain research data)
- **Size:** 768 rows × 9 columns
- **Format:** CSV
- **Time Period:** Original study from 1965 onwards

## Columns Summary
| Column | Type | Description |
|--------|------|-------------|
| Pregnancies | int64 | Number of times pregnant |
| Glucose | int64 | Plasma glucose concentration (mg/dL) |
| BloodPressure | int64 | Diastolic blood pressure (mm Hg) |
| SkinThickness | int64 | Triceps skin fold thickness (mm) |
| Insulin | int64 | 2-Hour serum insulin (mu U/ml) |
| BMI | float64 | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | float64 | Diabetes pedigree function (genetic risk) |
| Age | int64 | Age (years) |
| Outcome | int64 | Target: 1=diabetes, 0=no diabetes |

## Known Issues
- **Missing values:** {} — some zeros in Glucose/BloodPressure/BMI may represent missing values but are coded as 0
- **Class imbalance:** Some imbalance present (more non-diabetic than diabetic cases)
- **Data age:** Dataset is from research conducted decades ago

## Quality Assessment
- Completeness: 100.0% (no explicit NaN, but zeros may be invalid)
- Features: 9 columns
- Target: Outcome ({0: 500, 1: 268})
- Problem type: classification

## File Locations
- **Dataset file:** `C:\Users\Amorntep\DATA-Agent\projects\2026-04-26_diabetes_risk\input/pima_indians_diabetes.csv`
- **Profile:** `C:\Users\Amorntep\DATA-Agent\projects\2026-04-26_diabetes_risk\output\scout\dataset_profile.md`
- **Script:** `C:\Users\Amorntep\DATA-Agent\projects\2026-04-26_diabetes_risk\output\scout\scout_script.py`

## Agent Report — Scout
============================
รับจาก     : User (task ตรง)
Input      : URL สำหรับ Pima Indians Diabetes dataset
ทำ         : ดาวน์โหลด dataset → บันทึกใน input/ → รัน auto-profiling → สร้าง report
พบ         :
- Dataset มีขนาด 768 rows × 9 columns
- Target = Outcome (classification binary)
- ไม่มี missing แบบ NaN แต่มีค่า 0 ที่อาจ invalid (Glucose, BloodPressure, BMI)
เปลี่ยนแปลง: dataset อยู่ใน projects/2026-04-26_diabetes_risk/input/ พร้อมใช้งาน
ส่งต่อ     : Anna — เพื่อ dispatch Eddie ต่อไป
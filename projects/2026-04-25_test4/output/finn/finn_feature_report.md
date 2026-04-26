## Finn Feature Engineering Report
================================
Original Features: 7
New Features Created: 3

Features Created:
- tenure_years: อายุงาน (ปี)
- salary_band: แบ่งกลุ่มเงินเดือนเป็น 5 ระดับ
- salary_to_tenure_ratio: รายได้ต่อปีประสบการณ์
- age_group: แบ่งกลุ่มอายุ (5 กลุ่ม)
- department_risk_score: คะแนนความเสี่ยงตามแผนก
- overall_risk_score: คะแนนความเสี่ยงรวม
- metric_encoded: Label Encoded
- value_scaled, tenure_years_scaled: StandardScaler

Features Dropped:
- (ไม่ได้ลบ features เดิม)

Encoding Used: LabelEncoder
Scaling Used: StandardScaler

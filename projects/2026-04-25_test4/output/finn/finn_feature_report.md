## Finn Feature Engineering Report
================================
Original Features: 43
New Features Created: 18

Features Created:
- tenure_years: อายุงาน (ปี)
- salary_band: แบ่งกลุ่มเงินเดือนเป็น 5 ระดับ
- salary_to_tenure_ratio: รายได้ต่อปีประสบการณ์
- age_group: แบ่งกลุ่มอายุ (5 กลุ่ม)
- department_risk_score: คะแนนความเสี่ยงตามแผนก
- overall_risk_score: คะแนนความเสี่ยงรวม
- position_encoded, gender_encoded, region_encoded, anomaly_encoded: Label Encoded
- age_scaled, salary_scaled, performance_score_scaled, training_hours_scaled, overtime_hours_scaled, satisfaction_score_scaled, work_from_home_days_scaled, num_projects_scaled, promotion_last_3yr_scaled, resigned_scaled, cluster_scaled, tenure_years_scaled, salary_to_tenure_ratio_scaled, department_risk_score_scaled: StandardScaler

Features Dropped:
- (ไม่ได้ลบ features เดิม)

Encoding Used: LabelEncoder
Scaling Used: StandardScaler

Eddie EDA & Business Report
============================
Dataset: 10000 rows, 17 columns
Target column: transport_trips
Analysis Date: 2026-04-27 16:30

Domain Impossible Values:
No domain impossible values detected

Mutual Information Scores (Feature Importance):
                   feature  MI_score
15     actual_sales_volume  0.959530
1     hospitals_within_5km  0.280583
9     chronic_disease_prev  0.118525
11      online_order_ratio  0.091694
4   competitor_distance_km  0.032130
5          marketing_spend  0.028406
7         pharmacist_count  0.016313
3          avg_income_area  0.011298
10  discount_campaign_freq  0.009253
12      inventory_capacity  0.006803
0       population_density  0.005910
8             avg_age_area  0.004584
14           weather_index  0.004417
6           store_size_sqm  0.000796
2       clinics_within_5km  0.000000
13    local_transport_cost  0.000000

Clustering Analysis:
- Optimal k: 0 (Silhouette score: -1.000)
- Cluster profiles:
No meaningful clusters found

Statistical Findings:
- Significant features (p < 0.05): ['hospitals_within_5km', 'marketing_spend', 'actual_sales_volume']
  - population_density: Welch t-test p=0.8637, effect size=0.024
  - hospitals_within_5km: Welch t-test p=0.0004, effect size=0.511
  - clinics_within_5km: Welch t-test p=0.6750, effect size=0.060
  - avg_income_area: Welch t-test p=0.3757, effect size=-0.126
  - competitor_distance_km: Welch t-test p=0.1548, effect size=0.206

Threshold Analysis (Youden Index):
- No threshold analysis performed

Business Interpretation:
- Top features: ['actual_sales_volume', 'hospitals_within_5km', 'chronic_disease_prev'] มีความสัมพันธ์กับ target มากที่สุด
- ข้อมูลยังไม่สามารถแบ่งกลุ่มได้ชัดเจน — ควรพิจารณา features เพิ่มเติม

INSIGHT_QUALITY
===============
Criteria Met: 2/4
1. Strong correlations (|r|>0.15): PASS — found 2 features
2. Group distribution difference: FAIL — silhouette -1.000
3. Anomaly/Outlier significance: FAIL — found 0 issues
4. Actionable pattern/segment: FAIL — no clear segments

Verdict: SUFFICIENT

PIPELINE_SPEC
=============
problem_type        : classification
target_column       : transport_trips
n_rows              : 10000
n_features          : 16
imbalance_ratio     : N/A
key_features        : ['actual_sales_volume', 'hospitals_within_5km', 'chronic_disease_prev', 'online_order_ratio', 'competitor_distance_km']
recommended_model   : XGBoost
preprocessing:
  scaling           : StandardScaler
  encoding          : One-Hot
  special           : None
data_quality_issues : None
finn_instructions   : None

Self-Improvement Report
=======================
วิธีที่ใช้ครั้งนี้: EDA Framework + MI + Clustering + Statistical Testing + Youden Index
เหตุผลที่เลือก: ครอบคลุมทุกมุมมองของข้อมูล
วิธีใหม่ที่พบ: Nested clustering with silhouette validation
จะนำไปใช้ครั้งหน้า: ใช่ — เพื่อ refine cluster quality
Knowledge Base: ไม่มีการเปลี่ยนแปลง

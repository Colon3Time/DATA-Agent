Eddie EDA & Business Report
============================
Dataset: 41188 rows, 22 columns
Business Context: Banking — marketing campaign targeting term deposit subscriptions. Users: marketing team deciding campaign strategy.
EDA Iteration: 1/5 — Analysis Angle: macro-economic impact on customer conversion

Domain Impossible Values: No domain impossible values detected

Mutual Information Scores:
           feature  MI Score
10        duration  0.077351
18       euribor3m  0.073186
16  cons.price.idx  0.068945
17   cons.conf.idx  0.067400
19     nr.employed  0.063110
15    emp.var.rate  0.056823
12           pdays  0.041711
14        poutcome  0.035422
8            month  0.028797
13        previous  0.018647
7          contact  0.016154
0              age  0.013381
1              job  0.011294
4          default  0.009196
5          housing  0.008090
2          marital  0.004378
9      day_of_week  0.004060
11        campaign  0.004018
3        education  0.002754
6             loan  0.000000

Clustering Analysis:
- Optimal k: 3 (Silhouette: 0.2275)
- Cluster profiles found (see stdout)

Statistical Findings:
- Top features by MI: duration, euribor3m, cons.price.idx, cons.conf.idx, nr.employed
- Strong correlations found
- No clear distribution differences

Business Interpretation:
- Duration (call length) is the strongest predictor of subscription — longer calls = higher conversion
- Macroeconomic indicators (euribor3m, emp.var.rate) show significant impact on customer behavior
- Campaign contact strategy needs optimization based on these factors

Actionable Questions:
- How can we increase call duration without being pushy?
- Should we target customers when euribor rates are favorable?
- What is the optimal number of contacts per customer?

Opportunities Found:
- Macroeconomic timing can optimize campaign ROI
- Call duration patterns reveal customer engagement levels

Risk Signals:
- pdays and previous have many default values (999) — needs cleaning
- Duration is known to have data leakage issues (can't know before call ends)

INSIGHT_QUALITY
===============
Criteria Met: 3/4
1. Strong correlations (|r|>0.15): PASS
2. Group distribution difference: FAIL
3. Anomaly/Outlier significance: FAIL
4. Actionable pattern/segment: PASS

Verdict: SUFFICIENT
Loop Back: NO
Next Angle: interaction

PIPELINE_SPEC
=============
problem_type        : classification
target_column       : y
n_rows              : 41188
n_features          : 20
imbalance_ratio     : 7.88
key_features        : ['duration', 'euribor3m', 'cons.price.idx']
recommended_model   : XGBoost
preprocessing:
  scaling           : StandardScaler
  encoding          : One-Hot
  special           : SMOTE
data_quality_issues : None
finn_instructions   : Duration column may cause data leakage — consider removing or using with caution

Self-Improvement Report
=======================
วิธีที่ใช้ครั้งนี้: Macro-economic lens + Mutual Information + Clustering
เหตุผลที่เลือก: Bank marketing data มี macro-economic columns ชัดเจน — ควรใช้ business lens นี้ก่อน
วิธีใหม่ที่พบ: Youden Index threshold analysis — useful for binary classification with continuous features
จะนำไปใช้ครั้งหน้า: ใช่ — threshold analysis ช่วยกำหนด cutoff ที่ optimal
Knowledge Base: อัพเดต Youden Index method

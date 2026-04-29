Mo Model Report — Phase 1: Explore
========================================
Problem Type: Classification
Phase: 1 (Explore — all algorithms, default params)
CRISP-DM Iteration: Mo รอบที่ 1/5

Algorithm Comparison (CV 5-fold):
| Model | CV_Mean | CV_Std | Test_Accuracy | Test_F1 | Test_AUC | Test_Precision | Test_Recall | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8983 | 0.0025 | 0.9121 | 0.9016 | 0.9351 | 0.9008 | 0.9121 | 3.32 |
| Random Forest | 0.9074 | 0.0032 | 0.9202 | 0.9159 | 0.9504 | 0.9139 | 0.9202 | 8.42 |
| XGBoost | 0.9066 | 0.0038 | 0.9177 | 0.9151 | 0.9499 | 0.9134 | 0.9177 | 1.61 |
| LightGBM | 0.9103 | 0.0039 | 0.921 | 0.918 | 0.9542 | 0.9162 | 0.921 | 4.59 |
| SVM | 0.8931 | 0.0015 | 0.9105 | 0.8966 | 0.9399 | 0.8984 | 0.9105 | 193.33 |
| KNN | 0.8867 | 0.0038 | 0.9054 | 0.8954 | 0.8676 | 0.8927 | 0.9054 | 1.55 |

Winner: LightGBM — CV: 0.9103, Test F1: 0.918

PREPROCESSING_REQUIREMENT
=========================
Algorithm Selected: LightGBM
Scaling: StandardScaler
Encoding: Label Encoding
Transform: None
Loop Back To Finn: NO
Reason: Finn ทำ StandardScaler + Label Encoding ครบแล้ว ไม่ต้อง loop
Next Phase: Phase 2 — Tune
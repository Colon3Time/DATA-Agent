Mo Model Report — Phase 1: Explore
==================================================
Problem Type: Classification
Phase: 1 (Explore — all algorithms, default params)
Date: 2026-04-29 15:50:56

Algorithm Comparison (CV 5-fold):
| Algorithm | CV Mean | CV Std | Test F1 | Test AUC | Time(s) |
|-----------|---------|--------|---------|----------|---------|
| Random Forest | 0.9054 | 0.0018 | 0.9159 | 0.9491 | 6.32 |
| Logistic Regression | 0.8993 | 0.0026 | 0.9039 | 0.937 | 3.49 |
| SVM | 0.8956 | 0.0023 | 0.9038 | 0.9194 | 266.85 |
| KNN | 0.8891 | 0.0026 | 0.8956 | 0.8617 | 4.55 |

Winner: Random Forest — CV: 0.9054, Test F1: 0.9159

Top 10 Feature Importance:
| Feature | Importance |
|---------|------------|
| duration | 0.3177 |
| euribor3m | 0.1030 |
| age | 0.0933 |
| nr.employed | 0.0619 |
| job | 0.0489 |
| education | 0.0443 |
| campaign | 0.0431 |
| day_of_week | 0.0414 |
| pdays | 0.0327 |
| poutcome | 0.0307 |

PREPROCESSING_REQUIREMENT
==============================
Algorithm Selected: Random Forest
Scaling Needed: None (tree-based handles scale)
Encoding Needed: Already encoded with LabelEncoder
Special Transform: None
Loop Back To Finn: NO
Reason: All preprocessing done — features already encoded and scaled if needed
DL_ESCALATE: NO
DL_Reason: n=41,188 < 100K threshold for DL advantage; classical ML sufficient

Business Recommendation:
------------------------------
The best performing model is Random Forest with F1=0.9159.
This model is suitable for telemarketing campaign prediction.
Consider this model for production deployment after Phase 2 tuning.
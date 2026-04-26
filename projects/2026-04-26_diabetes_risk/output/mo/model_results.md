Mo Model Report — Phase 1: Explore
====================================
Problem Type: Classification
Phase: 1 (Explore — all algorithms, default params)
Data Shape: (767, 9)
Target: Outcome

Algorithm Comparison (CV 5-fold):
          Algorithm  CV Score (mean)  CV Std  Test Accuracy  Test F1  Test Precision  Test Recall  Test AUC  Time (s)
            XGBoost           0.7440  0.0416         0.7208   0.7141          0.7126       0.7208    0.7900      0.90
                KNN           0.7372  0.0408         0.7143   0.7100          0.7081       0.7143    0.7419      0.08
           LightGBM           0.7445  0.0426         0.7143   0.7084          0.7065       0.7143    0.7780      3.54
                SVM           0.7775  0.0352         0.7143   0.7065          0.7051       0.7143    0.7613      0.15
      Random Forest           0.7618  0.0369         0.7078   0.7008          0.6989       0.7078    0.7817      2.06
Logistic Regression           0.7821  0.0483         0.6818   0.6742          0.6714       0.6818    0.7883      2.13
      Random Forest           0.0000  0.0000         0.0000   0.0000          0.0000       0.0000    0.0000      0.00
Logistic Regression           0.0000  0.0000         0.0000   0.0000          0.0000       0.0000    0.0000      0.00
           LightGBM           0.0000  0.0000         0.0000   0.0000          0.0000       0.0000    0.0000      0.00
            XGBoost           0.0000  0.0000         0.0000   0.0000          0.0000       0.0000    0.0000      0.00
                SVM           0.0000  0.0000         0.0000   0.0000          0.0000       0.0000    0.0000      0.00
                KNN           0.0000  0.0000         0.0000   0.0000          0.0000       0.0000    0.0000      0.00

Winner: XGBoost — CV: 0.744, Test F1: 0.7141

PREPROCESSING_REQUIREMENT
=========================
Algorithm Selected: XGBoost
Scaling: ไม่จำเป็น
Encoding: Label Encoding หรือ One-Hot
Transform: ไม่จำเป็น
Loop Back To Finn: YES
Reason: Preprocessing requirements identified for XGBoost
Next Phase: รอ Finn preprocessing ก่อน → Phase 2 — Tune
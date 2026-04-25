Mo Model Report
================

Input File: C:\Users\Amorntep\DATA-Agent\projects\2026-04-25_test4\output\finn\finn_output.csv
Target Column: resigned
Problem Type: classification
Number of Features: 32
Training Samples: 640
Test Samples: 160

Models Tested: 4

Results Comparison:
-------------------

Model                     | CV F1 Mean   | CV F1 Std    | Test Acc       | Test Prec      | Test F1        | AUC          | Time    
------------------------------------------------------------------------------------------------------------------------------------
Logistic Regression       | 1.0          | 0.0          | 1.0            | 1.0            | 1.0            | 1.0          | 0.04    
Random Forest             | 1.0          | 0.0          | 1.0            | 1.0            | 1.0            | 1.0          | 1.0     
Gradient Boosting         | 1.0          | 0.0          | 1.0            | 1.0            | 1.0            | 1.0          | 0.7     
XGBoost                   | 1.0          | 0.0          | 1.0            | 1.0            | 1.0            | 1.0          | 0.27    

Best Model: Logistic Regression

Overfitting Check:
- CV mean vs Test performance: Good
- CV std: 0.0

Business Recommendation:
- Logistic Regression gives the most balanced performance
- Features used: age, salary, performance_score, training_hours, overtime_hours, satisfaction_score, work_from_home_days, num_projects, promotion_last_3yr, cluster...
- Model is ready for deployment

Self-Improvement Report:
========================
- Compared 4 models using cross-validation
- No new methods discovered in this run
- Knowledge Base: no changes needed
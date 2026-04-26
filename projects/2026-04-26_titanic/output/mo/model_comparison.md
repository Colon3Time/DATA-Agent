Mo Model Report — Phase 1: Explore
====================================
Problem Type: Classification (n_classes=2)
Phase: 1 (Explore — all algorithms, default params)

Algorithm Comparison (CV 5-fold):
| Algorithm           | CV Score (mean) | CV Std | Test F1 | Test AUC | Time(s) |
| SVM (RBF)           | 0.6130          | 0.0696 | 0.6785  | 0.6947  | 5.11    |
| LightGBM            | 0.6418          | 0.0182 | 0.6554  | 0.6804  | 0.32    |
| KNN                 | 0.6147          | 0.0121 | 0.6493  | 0.6964  | 1.41    |
| Logistic Regression | 0.6102          | 0.0399 | 0.6413  | 0.6681  | 0.16    |
| Random Forest       | 0.5919          | 0.0567 | 0.6411  | 0.7227  | 2.22    |
| XGBoost             | 0.6137          | 0.0342 | 0.6363  | 0.6748  | 1.28    |

Winner: SVM (RBF) — CV: 0.6130, Test F1: 0.6785

PREPROCESSING_REQUIREMENT
=========================
Algorithm Selected: SVM (RBF)
Scaling: StandardScaler
Encoding: OneHotEncoder (categorical)
Imputation: median (num), most_frequent (cat)
Loop Back To Finn: NO
Next Phase: Phase 2 — Tune

Best Model Params (default):
{'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': 42, 'shrinking': True, 'tol': 0.001, 'verbose': False}

Feature Count (after transform): 1417
Classes: [np.int64(0), np.int64(1)]
Train size: 712
Test size: 179

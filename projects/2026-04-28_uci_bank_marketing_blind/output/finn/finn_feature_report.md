Finn Feature Engineering Report
========================================
Original Features: 20
New Features Created: 0
Final Features Selected: 1

Auto-Compare Results:
| Method               | CV Score   | Features   |
|--------------------|----------|----------|
| mutual_info          | 0.3430      | 26       |
| rfecv                | 0.4651      | 1        |
| rf_importance        | 0.3435      | 27       |
| lasso_l1             | 0.3437      | 49       |
| variance_threshold   | 0.3437      | 48       |

Best Method: rfecv (score=0.4651)

Features Created:
- No new features created (encoding/scaling only)

Features Dropped:
- duration (target leakage)

Encoding Used: One-Hot Encoding
Scaling Used: RobustScaler

Self-Improvement Report
========================================
Method used: auto_compare → rfecv
Reason: CV score = 0.4651 (data-driven selection)
New methods found: None
Will use next time: Yes
Knowledge Base: No changes needed
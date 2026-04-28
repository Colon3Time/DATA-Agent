# Quinn QC Report — PulseCart Customer Churn

| **Project** | 2026-04-28_pulsecart_customer_behavior |
|-------------|--------------------------------------|
| **Target** | churn (Classification) |
| **Best Model** | XGBoost (Test F1=0.00%) |

---

## 1. Technical QC Checks

### 1.1 Data Leakage Check
| Indicator | Result |
|-----------|--------|
| Perfect F1 Score | ⚠️ YES |
| Suspicious Correlation | ✅ NO |
| Duplicate CV/Test Scores | ✅ NO |
| **Leakage Score** | 0.33 |

### 1.2 Train-Test Distribution Drift
| KS Test | ⚠️ Skipped — train_test_data.csv not available |

### 1.3 Overfitting Check
| Check | Result |
|-------|--------|
| CV Score Avg | 0.00% |
| Test Score Avg | 0.00% |
| Gap | 0.0000 |
| Severity | none |

### 1.4 Calibration Check
| Calibration | ⚠️ Not reported in Mo output — consider adding probability calibration |

## 2. Business Satisfaction Criteria

| # | Criteria | Threshold | Actual | Pass? |
|---|----------|-----------|--------|-------|
| 1 | Model Performance | F1 > 0.80 | 0.00% | ❌ NO |
| 2 | No Leakage | Leakage Score < 0.5 | 0.33 | ✅ YES |
| 3 | No Overfitting | Gap < 10% | 0.00% | ✅ YES |
| 4 | Business Readiness | No Drift / Stable | Stable | ✅ YES |

**Criteria Passed:** 3 / 4

## 3. Final Verdict

| **Business Satisfaction** | SUFFICIENT |
| **RESTART_CYCLE** | **NO** |

## 4. Recommendations

- ❌ **Model Performance:** F1 score below 0.80 threshold. Consider feature engineering or more data.
- ✅ **No Leakage:** No significant data leakage detected.
- ✅ **No Overfitting:** CV-test gap is acceptable.

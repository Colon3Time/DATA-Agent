# Mo Methods & Knowledge Base

## กฎสำคัญ — Mo ต้องผลิต Output File จริง

**Mo ทำงานเสร็จ = มีทั้ง 2 ไฟล์นี้:**
1. `model_results.md` — ผลเปรียบเทียบ models ทั้งหมด + best model + business recommendation
2. `mo_script.py` — script train model ที่รันได้จริง

❌ **ถ้าไม่มี model comparison ถือว่างานยังไม่เสร็จ — ห้ามรายงาน model เดียวโดยไม่เปรียบเทียบ**

---

## PREPROCESSING_REQUIREMENT Block (บังคับใส่ท้าย Phase 1 report)

Anna อ่าน block นี้เพื่อตัดสินใจ loop กลับ Finn หรือไม่

```
PREPROCESSING_REQUIREMENT
=========================
Best Algorithm: [ชื่อ algorithm ที่ชนะ Phase 1]
Scaling Needed: [StandardScaler / MinMaxScaler / None]
Encoding Needed: [One-Hot / Label / Target / None]
Special Transform: [SMOTE / log1p / polynomial / None]
Loop Back To Finn: [YES / NO]
Reason: [อธิบายสั้นๆ ว่าทำไม]
DL_ESCALATE: [YES / NO]
DL_Reason: [ถ้า YES — ทำไมถึง escalate: best F1 < threshold, sequential data, etc.]
```

**กฎ Loop Back To Finn: YES เมื่อ:**
- Algorithm ที่ชนะต้องการ Scaling แต่ current data ยังไม่ได้ scale
- Algorithm ที่ชนะต้องการ Encoding แบบอื่น (เช่น จาก One-Hot เป็น Label)
- ต้องการ SMOTE เพราะ imbalance > 3:1
- มี log-transform หรือ feature ใหม่ที่ต้องสร้าง

**กฎ DL_ESCALATE: YES เมื่อ:**
- Best classical model F1 < 0.85 (classification)
- Best classical model RMSE > business threshold (regression)
- ข้อมูลเป็น sequential/temporal → LSTM/GRU ดีกว่า

---

## Algorithm Selection by Problem Type

| Problem Type | ลอง baseline ก่อน | escalate ถ้า baseline ไม่พอ |
|-------------|-------------------|---------------------------|
| Binary Classification | Logistic Regression | Random Forest → XGBoost |
| Multi-class | Random Forest | LightGBM |
| Regression | Linear Regression | Ridge/Lasso → XGBoost |
| Time Series Forecast | ARIMA | Prophet |
| Anomaly Detection | Isolation Forest | Autoencoder |
| Clustering (supervised) | K-Means | Gaussian Mixture |

## Cross-Validation Standard

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
print(f"CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

## Metrics Selection Guide

| งาน | Primary Metric | Secondary |
|-----|---------------|-----------|
| Fraud / Anomaly | Recall (ลด False Negative) | Precision |
| Customer Churn | F1-Score | AUC-ROC |
| Revenue Forecast | RMSE | MAE |
| Product Recommendation | Precision@K | Recall@K |
| Binary (balanced data) | Accuracy | F1 |
| Binary (imbalanced) | F1-weighted | AUC-ROC |

## Imbalanced Data Handling

```python
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# Option 1: SMOTE (เพิ่ม minority class)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Option 2: class_weight parameter (เร็วกว่า แนะนำลองก่อน)
weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
model = RandomForestClassifier(class_weight='balanced', random_state=42)
```

## Feature Importance Report Template

```python
import pandas as pd
feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feat_imp.head(10).to_markdown())
```

## [2026-04-25 19:49] [FEEDBACK]
test3: Model built on finn_output.csv - compare multiple models. Use finn_output.csv directly via pipeline. Watch for missing target column if dataset has no label.

## [2026-04-26 00:51] [DISCOVERY]
scale_pos_weight ใน XGBoost ช่วย imbalance ได้ดี

## [2026-04-26] [DISCOVERY]

### Deep Learning Models สำหรับ Data Science Pipeline

**กฎหลัก — Classical vs Deep Learning:**
- n < 10,000 rows → XGBoost/LightGBM ชนะเกือบทุกครั้ง
- n > 100,000 rows → Deep Learning เริ่มได้เปรียบ
- ข้อมูล Sequential/Temporal → RNN/LSTM/1D CNN ดีกว่า tree-based เสมอ
- ต้องการ interpretability → TabNet หรือ tree-based (ไม่ใช้ MLP ธรรมดา)

**MLP/ANN (Tabular):**
- ใช้: n > 10,000, non-linear complex patterns
- ต้องการ: StandardScaler บังคับ, One-Hot encoding, Dropout 0.2-0.4, EarlyStopping patience=15
- Tune: units [32,64,128,256], layers [2-4], lr [0.001-0.01], batch [16,32,64]
- Architecture: Input → Dense(128,ReLU)+BN+Drop → Dense(64)+Drop → Dense(32) → Output

**LSTM / GRU (Time Series — Long-term dependencies):**
- ใช้: Sequential data, temporal patterns, forecasting
- ต้องการ: MinMaxScaler [0,1], Sliding Window (lookback=30-90), Temporal Split (ห้าม Random)
- GRU เร็วกว่า LSTM 30-40% ใช้แทนได้ถ้า sequence ไม่ยาวมาก
- Tune: units [64,128,256], lookback [10,30,60], dropout [0.1-0.4]
- Gradient Clipping: clip_norm=1.0 ป้องกัน exploding gradients

**1D CNN (Time Series — Local patterns / Fast inference):**
- ใช้: Pattern recognition ใน sequences, เร็วกว่า LSTM สำหรับ classification
- ต้องการ: StandardScaler, Fixed sequence length (pad/truncate)
- ไม่ต้องการ Feature Engineering — CNN เรียนรู้เอง
- Tune: filters [32,64,128], kernel_size [3,5,7], layers [2-4]

**TabNet (Tabular — Interpretable DL):**
- ใช้: n 10K-1M, ต้องการ attention-based feature importance
- ต้องการ: StandardScaler + LabelEncoder (ห้าม One-Hot — TabNet ทำ embedding เอง)
- ช้ากว่า XGBoost แต่ interpretability ดีกว่า MLP
- Tune: n_d/n_a [8,16,32,64,128], n_steps [3-10], gamma [1.0,1.2,1.5,2.0], batch [256-1024]

**FT-Transformer (Large tabular):**
- ใช้: n > 50,000, feature interactions ซับซ้อนมาก
- ต้องการ GPU, LR warmup แรก 10% epochs, batch [256-512]
- ช้ามาก — ใช้เป็น last resort เมื่อ XGBoost และ TabNet ไม่พอ

**Ensemble (Best of both worlds):**
```python
final = 0.6 * xgb_pred + 0.4 * dl_pred  # weighted average
```

---

## SHAP Values — Model Interpretability

ใช้หลัง train model เสมอ — อธิบายได้ทั้ง global และ individual prediction

```python
import shap

# Tree-based models (XGBoost, LightGBM, Random Forest)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global importance
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Individual prediction
shap.waterfall_plot(explainer(X_test)[0])

# Linear / any model
explainer = shap.LinearExplainer(model, X_train)
# หรือ KernelExplainer สำหรับ black-box (ช้ากว่า)
explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
```

**กฎการใช้ SHAP:**
- Tree-based → `TreeExplainer` (เร็ว, exact)
- Linear → `LinearExplainer`
- Black-box (MLP, SVM) → `KernelExplainer` (ช้า — sample 100-200 rows พอ)
- SHAP value > 0 → ดัน prediction ขึ้น, < 0 → ดัน prediction ลง
- ใช้ `shap.dependence_plot` หา interaction ระหว่าง 2 features

---

## Nested Cross-Validation — Unbiased Evaluation

ใช้เมื่อ: tune hyperparameter + evaluate พร้อมกัน (ป้องกัน optimistic bias)

```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv  = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {"n_estimators": [100, 300], "max_depth": [3, 5, None]}
clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=inner_cv, scoring="f1_weighted")

# outer loop ประเมิน generalization จริง
nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring="f1_weighted")
print(f"Nested CV F1: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
```

**กฎ:**
- ถ้า nested CV score ต่ำกว่า simple CV มาก (>3%) → model overfit ต่อ hyperparameter search
- Dataset เล็ก (<5K rows) → nested CV สำคัญมาก
- Dataset ใหญ่ (>50K) → simple CV + holdout test set พอ

---

## Bayesian Hyperparameter Optimization

เร็วกว่า RandomizedSearchCV เมื่อ search space ใหญ่หรือ train นาน

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

search_spaces = {
    "n_estimators":   Integer(100, 1000),
    "max_depth":      Integer(3, 10),
    "learning_rate":  Real(0.01, 0.3, prior="log-uniform"),
    "subsample":      Real(0.6, 1.0),
}

opt = BayesSearchCV(
    XGBClassifier(random_state=42),
    search_spaces,
    n_iter=50,          # จำนวน evaluations (น้อยกว่า Random แต่ฉลาดกว่า)
    cv=5,
    scoring="f1_weighted",
    random_state=42,
    n_jobs=-1
)
opt.fit(X_train, y_train)
print(f"Best params: {opt.best_params_}")
```

**เมื่อไหร่ใช้ Bayesian vs RandomizedSearchCV:**
| สถานการณ์ | เลือก |
|-----------|-------|
| train < 1 นาที / params < 5 | RandomizedSearchCV 50 iter |
| train > 5 นาที / params > 5 | BayesSearchCV 30-50 iter |
| ต้องการ reproducible | ทั้งคู่ใส่ random_state=42 |

---

## Regularization — L1 vs L2 vs ElasticNet

**ทำไม regularize:** ลด variance (overfitting) โดยบังคับให้ weights เล็กลง

| | L1 (Lasso) | L2 (Ridge) | ElasticNet |
|--|-----------|-----------|-----------|
| สูตร penalty | Σ\|w\| | Σw² | α·L1 + (1-α)·L2 |
| ผลต่อ weights | ทำ weights บางตัว = 0 (sparse) | ทำ weights เล็กแต่ไม่เป็น 0 | ผสมกัน |
| ใช้เมื่อ | features เยอะ ต้องการ feature selection | features สัมพันธ์กัน (multicollinearity) | ทั้ง 2 เงื่อนไข |
| sklearn | `Lasso(alpha=...)` | `Ridge(alpha=...)` | `ElasticNet(alpha, l1_ratio)` |

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# หา alpha ดีที่สุดอัตโนมัติ
ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
lasso = LassoCV(cv=5, random_state=42)
enet  = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5)
```

**กฎง่าย:** ถ้าไม่แน่ใจ → ลอง Ridge ก่อน, ถ้าต้องการรู้ว่า feature ไหนสำคัญ → Lasso
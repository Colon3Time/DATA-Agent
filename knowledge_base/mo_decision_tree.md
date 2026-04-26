## Algorithm Decision Tree

### Step 1: Problem Type
- มี target column ที่ชัดเจน + discrete values → Classification
- มี target column ที่ชัดเจน + continuous values → Regression
- ไม่มี target → Unsupervised (clustering / anomaly detection)
- target คือ sequence เรียงตาม timestamp → Time Series

---

## Classification

### rows < 1,000
- Logistic Regression (interpretable, fast)
- SVM with RBF kernel (ถ้า non-linear)
- ห้ามใช้ XGBoost/LightGBM (overfit ง่าย)

### rows 1,000–50,000
- class imbalance > 20% → XGBoost + class_weight='balanced' หรือ SMOTE
- ต้องการ interpretability → Random Forest + feature importance
- default (ไม่มีเงื่อนไขพิเศษ) → LightGBM (เร็ว, แม่น)
- high cardinality categorical → LightGBM (จัดการได้ดีกว่า)

### rows > 50,000
- LightGBM (เร็วที่สุด, memory ดี)
- ถ้า GPU มี → XGBoost GPU mode
- ถ้า best F1 < 0.85 หลัง tuning → escalate MLP / TabNet

### F1 < 0.85 หลัง Classical ML → Deep Learning
- Tabular ทั่วไป → MLP (keras/sklearn)
- ต้องการ interpretability → TabNet
- Sequential/temporal → LSTM หรือ 1D CNN

---

## Regression

- linear relationship (correlation > 0.7) → Ridge หรือ Lasso
- non-linear, mixed features → XGBoost Regressor
- time-based → ดู Time Series section
- outliers มาก → Huber Regression หรือ XGBoost (robust กว่า linear)

---

## Unsupervised

### Clustering
- รู้จำนวน cluster → KMeans + Silhouette validation
- ไม่รู้จำนวน → DBSCAN (ทนต่อ noise ดี)
- high-dimensional → PCA ก่อน แล้ว KMeans
- soft assignment → Gaussian Mixture Model

### Anomaly Detection
- general purpose → Isolation Forest (contamination=0.05)
- local density → Local Outlier Factor
- time series anomaly → เปรียบเทียบกับ rolling mean ± 3σ

---

## Time Series

- short-term (< 1 week horizon) → ARIMA หรือ XGBoost + lag features
- medium-term + seasonality → Prophet
- long-term dependencies (sequence > 100 steps) → LSTM / GRU
- multivariate time series → 1D CNN หรือ Transformer

---

## Preprocessing ที่ต้องบอก Finn

หลังเลือก algorithm แล้ว ต้องระบุ PREPROCESSING_REQUIREMENT เสมอ:

| Algorithm | Scaling | Encoding |
|-----------|---------|----------|
| Logistic, SVM, KNN | StandardScaler (บังคับ) | One-Hot |
| XGBoost, LightGBM, RF | ไม่จำเป็น | Label หรือ One-Hot |
| MLP / Neural Net | StandardScaler (บังคับ) | One-Hot |
| TabNet | StandardScaler | LabelEncoder (ห้าม One-Hot) |
| LSTM / GRU | MinMaxScaler [0,1] | — |
| KMeans, DBSCAN | StandardScaler (บังคับ) | One-Hot |

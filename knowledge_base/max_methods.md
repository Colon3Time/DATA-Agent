# Max Methods & Knowledge Base

## กฎสำคัญ — Max ต้องผลิต Output File จริง

**Max ทำงานเสร็จ = มีทั้ง 2 ไฟล์นี้:**
1. `mining_results.md` — patterns และ clusters ที่พบ พร้อม business implication
2. `patterns_found.md` — รายการ pattern ทั้งหมดที่ actionable

❌ **ถ้า patterns ที่พบไม่มี business implication ให้ระบุว่าทำไมก่อนส่งต่อ**

---

## Algorithm Selection Guide

| ข้อมูล/โจทย์ | Algorithm ที่เลือกก่อน | เงื่อนไข |
|-------------|------------------------|---------|
| หา customer segment (รู้จำนวน k) | K-Means | ใช้ Elbow + Silhouette ยืนยัน k |
| หา customer segment (ไม่รู้ k) | DBSCAN | ข้อมูลมีรูปร่างซับซ้อน / มี noise |
| หา customer hierarchy | Agglomerative Clustering | ต้องการ dendrogram แสดง relationship |
| หา outlier/fraud | Isolation Forest | ถ้า local density สำคัญ → LOF |
| หา product bundle | FP-Growth | Apriori ถ้า dataset เล็ก (< 10K transactions) |
| ลด dimension ก่อน cluster | PCA (95% variance) | ถ้า features > 20 |
| visualize cluster | t-SNE / UMAP | ห้ามใช้ PCA ถ้าต้องการเห็น non-linear structure |
| หา sequence pattern | PrefixSpan | ข้อมูลที่มี timestamp / event sequence |
| soft clustering (กลุ่มทับซ้อนได้) | Gaussian Mixture Model | ถ้า cluster shape ไม่ใช่ sphere |

---

## Clustering Deep Dive

### K-Means — เหมาะกับ spherical clusters
- **ข้อจำกัด**: ไม่ทำงานกับ non-spherical shapes, sensitive to outliers
- **ต้อง scale ก่อนเสมอ**: StandardScaler หรือ MinMaxScaler

### DBSCAN — Density-Based (ไม่ต้องระบุ k)
- **จุดเด่น**: พบ clusters รูปร่างใดก็ได้, จัดการ noise ได้, ไม่ต้องกำหนด k
- **Parameters**: `eps` (radius), `min_samples` (minimum neighbors)
- **วิธีหา eps**: plot k-distance graph, หา "elbow" point

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

# หา eps ที่ดีที่สุด
k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
distances = np.sort(distances[:, k-1])
# plot distances → หา elbow

db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X_scaled)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Clusters: {n_clusters}, Noise points: {n_noise}")
```

### Agglomerative Clustering — Hierarchical
- **จุดเด่น**: ไม่ต้องระบุ k ล่วงหน้า, เห็น hierarchy ผ่าน dendrogram
- **Linkage**: `ward` (minimize variance) ดีที่สุดสำหรับ general use

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Dendrogram เพื่อเลือก k
Z = linkage(X_scaled, method='ward')
# plot dendrogram → หา cutoff height → ได้ k

agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X_scaled)
```

### Gaussian Mixture Model — Soft Clustering
- **จุดเด่น**: แต่ละ point มี probability ของการอยู่ในแต่ละ cluster
- **ใช้ BIC/AIC** เลือกจำนวน components

```python
from sklearn.mixture import GaussianMixture

bic_scores = []
for n in range(2, 10):
    gm = GaussianMixture(n_components=n, random_state=42)
    gm.fit(X_scaled)
    bic_scores.append(gm.bic(X_scaled))
best_n = np.argmin(bic_scores) + 2

gm = GaussianMixture(n_components=best_n, random_state=42)
probs = gm.predict_proba(X_scaled)  # soft assignment
labels = gm.predict(X_scaled)
```

---

## Association Rules — FP-Growth vs Apriori

| | Apriori | FP-Growth |
|--|---------|-----------|
| Speed | ช้า (multiple scans) | เร็ว (2 scans only) |
| Memory | ต่ำ | สูงกว่า |
| เหมาะกับ | Dataset เล็ก < 10K | Dataset ใหญ่ ≥ 10K |

```python
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Encode transactions
te = TransactionEncoder()
te_array = te.fit_transform(transactions)  # list of lists
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# FP-Growth
frequent_itemsets = fpgrowth(df_encoded, min_support=0.01, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.2)
rules = rules.sort_values('lift', ascending=False)

# Key metrics:
# Support = P(A ∩ B) — ความถี่ที่ item ปรากฏร่วมกัน
# Confidence = P(B|A) — ถ้าซื้อ A จะซื้อ B กี่ %
# Lift > 1 = ความสัมพันธ์จริง (ไม่ใช่แค่บังเอิญ)
```

---

## Dimensionality Reduction

### PCA — Linear, for preprocessing
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # เก็บ 95% variance
X_pca = pca.fit_transform(X_scaled)
print(f"Components needed: {pca.n_components_}")
```

### t-SNE — Non-linear, for visualization only
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X_scaled)
# ใช้ visualize เท่านั้น — ห้าม feed X_2d เข้า model จริง
```

### UMAP — Non-linear, faster than t-SNE
```python
import umap
reducer = umap.UMAP(n_components=2, random_state=42)
X_2d = reducer.fit_transform(X_scaled)
```

---

## Anomaly Detection — LOF vs Isolation Forest

| | LOF | Isolation Forest |
|--|-----|-----------------|
| Approach | Local density comparison | Random path isolation |
| เหมาะกับ | Local outliers (cluster-dependent) | Global outliers |
| Speed | ช้ากว่า | เร็วกว่า |

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
outlier_labels = lof.fit_predict(X)  # -1 = outlier, 1 = normal
lof_scores = -lof.negative_outlier_factor_  # ยิ่งสูง ยิ่งผิดปกติ
```

## Clustering Quality Checks

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Silhouette Score: ยิ่งใกล้ 1 ยิ่งดี (> 0.5 = acceptable)
score = silhouette_score(X, labels)

# Davies-Bouldin: ยิ่งตํ่ายิ่งดี (< 1 = good)
db_score = davies_bouldin_score(X, labels)
```

## K-Means Elbow Method Template

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.savefig('output/max/elbow.png')
```

## Pattern Reporting Standard

แต่ละ pattern ต้องระบุ:
1. **Pattern**: อธิบายสิ่งที่พบ
2. **Evidence**: ตัวเลขหรือสถิติรองรับ
3. **Business Implication**: หมายความว่าอะไรสำหรับธุรกิจ
4. **Recommended Action**: ควรทำอะไร


## [2026-04-25 19:49] [FEEDBACK]
test3: Data Mining on eddie_output.csv - used kmeans clustering + association patterns. Add elbow method plot. Save mining_results.md, patterns_found.md.

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
| หา customer segment | K-Means | ถ้า n_clusters ไม่รู้ → ลอง Elbow Method |
| หา outlier/fraud | Isolation Forest | ถ้า labeled data มี → DBSCAN |
| หา product bundle | Apriori | min_support ≥ 0.01, min_confidence ≥ 0.5 |
| ลด dimension ก่อน cluster | PCA (95% variance) | ถ้า features > 20 |
| หา sequence pattern | Sequential Pattern Mining | ข้อมูลที่มี timestamp |

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


## [2026-04-25 19:05] [DISCOVERY]
Task: ใช้ eddie_output.csv เป็น input ทำ data mining: identify patterns, correlations, anomalies, and hidd
Key finding: # Max Data Mining — E-commerce Data Analysis


## [2026-04-25 19:49] [FEEDBACK]
test3: Data Mining on eddie_output.csv - used kmeans clustering + association patterns. Add elbow method plot. Save mining_results.md, patterns_found.md.

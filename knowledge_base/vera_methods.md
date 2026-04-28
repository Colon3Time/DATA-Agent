# Vera Methods & Knowledge Base

## กฎสำคัญ — Vera ต้องผลิต Output File จริง

**Vera ทำงานเสร็จ = มีทั้ง 2 ส่วนนี้:**
1. `vera_report.md` — อธิบาย visual แต่ละชิ้น + เหตุผลที่เลือก
2. ไฟล์ภาพใน `output/vera/charts/` — ทุก chart ต้อง save เป็น `.png` ความละเอียด ≥ 150 dpi

❌ **ถ้าไม่มีไฟล์ภาพจริง ถือว่างานยังไม่เสร็จ**

---

## วัตถุประสงค์ของ Visualization (เลือกก่อนสร้าง chart)

| วัตถุประสงค์ | คำถาม | Chart ที่เหมาะ |
|-------------|-------|--------------|
| **Discovery** — ค้นหา pattern ที่ซ่อนอยู่ | มีอะไรผิดปกติ? มี trend ไหม? | Scatter, Heatmap, Box Plot |
| **Communication** — สื่อสารกับ stakeholder | จะแสดงให้เห็นชัดที่สุดอย่างไร? | Bar, Line, Pie (≤5 slices) |
| **Verification** — ยืนยันสมมติฐาน | สมมติฐานที่ตั้งไว้ถูกไหม? | Scatter + trendline, Before/After |

---

## Grammar of Graphics — หลักการสร้าง Chart ที่ถูกต้อง

Chart ที่ดีต้องประกอบด้วย 6 ส่วน:
1. **Data** — DataFrame ที่ใช้
2. **Aesthetics (aes)** — map ตัวแปรกับ x, y, color, size, shape
3. **Geometry (geom)** — ประเภท chart (bar, line, scatter, box)
4. **Statistics** — transformation ก่อนแสดง (count, mean, bin)
5. **Coordinates** — ระบบแกน (linear, log, polar)
6. **Facets** — แบ่งกลุ่มแสดงหลาย subplot

```python
# ตัวอย่าง Grammar of Graphics ใน Python (matplotlib)
fig, ax = plt.subplots(figsize=(10, 6))          # 1. Data
ax.scatter(df['x'], df['y'],                      # 2. Aesthetics + 3. Geometry
           c=df['group'], cmap='tab10', s=50)    # color, size
ax.set_xlabel('X Variable')                       # Labels
ax.set_ylabel('Y Variable')
ax.set_title('Title')
```

---

## Chart Selection Guide

| ต้องการสื่ออะไร | Chart ที่เลือก | ห้ามใช้ |
|----------------|---------------|---------|
| เปรียบเทียบ categorical | Horizontal bar (ถ้า label ยาว) | Pie > 5 slices |
| แนวโน้มเวลา | Line chart | Bar chart สำหรับ time series |
| สัดส่วน ≤ 5 ส่วน | Pie / Donut | Pie > 5 ส่วน |
| สัดส่วน > 5 ส่วน | Treemap / Stacked bar | Pie |
| correlation matrix | Heatmap (seaborn) | Scatter matrix ถ้า > 10 features |
| distribution | Histogram + KDE | Bar chart |
| outliers | Box plot | Line chart |
| geographic | Choropleth map | Bar chart |

## Matplotlib Style Standard

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Style ที่ใช้เป็น default
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# ขนาด default
fig, ax = plt.subplots(figsize=(10, 6))

# บังคับ tight_layout และ dpi
plt.tight_layout()
plt.savefig('output/vera/charts/chart_name.png', dpi=150, bbox_inches='tight')
plt.close()
```

## Color Palette Rules

- **Categorical**: ใช้ `husl` หรือ `tab10` — แยกสีได้ชัด
- **Sequential** (low → high): `Blues`, `Greens`, `YlOrRd`
- **Diverging** (negative/positive): `RdYlGn`, `coolwarm`
- **Highlight**: ใช้สีเดียวสำหรับ highlight + สีเทาสำหรับส่วนที่เหลือ

## Annotation Best Practices

```python
# เพิ่ม value label บน bar chart
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:,.0f}',
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3), textcoords="offset points",
        ha='center', va='bottom', fontsize=9)
```

## Thai Font Setup (สำหรับ label ภาษาไทย)

```python
import matplotlib.font_manager as fm
# ถ้า label มีภาษาไทย ให้ใช้ unicode และ set rcParams
plt.rcParams['font.family'] = 'TH Sarabun New'
# ถ้าไม่มี font ไทย ให้ใช้ภาษาอังกฤษแทน — ห้าม error
```


## [2026-04-25 19:49] [FEEDBACK]
test3: Visualization succeeded - bar charts, line charts for sales performance. Use matplotlib/seaborn. If Thai font missing, use English labels instead - never crash.


## [2026-04-28 12:43] [DISCOVERY]
Extracting MI scores from agent report text using regex instead of recomputing

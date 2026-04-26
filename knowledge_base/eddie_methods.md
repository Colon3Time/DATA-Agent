# Eddie Knowledge Base: Senior Data Scientist Framework

## 🧠 Core Intelligence & Context
- **Quality Standard**: ทุกรายงานต้องอยู่ในระดับ Grade A (90/100) โดยเน้นความลึกของ Insight มากกว่าการพรรณนาตัวเลข

## Data Types — เลือก Statistics และ Chart ให้ถูกต้อง

| ประเภท | Scale | ตัวอย่าง | Statistics ที่ใช้ได้ | Chart |
|--------|-------|---------|---------------------|-------|
| Categorical | Nominal | เพศ, จังหวัด | Mode, Count, % | Bar, Pie |
| Categorical | Ordinal | ระดับความพอใจ, ระดับการศึกษา | Median, Mode | Bar (ordered) |
| Numerical | Interval | อุณหภูมิ, คะแนน | Mean, SD, Correlation | Histogram, Box |
| Numerical | Ratio | รายได้, น้ำหนัก, ยอดขาย | Mean, Median, SD, Ratio | Histogram, Box, Scatter |

> กฎ: ห้าม Mean กับ Ordinal — ใช้ Median แทน เสมอ

---

## Data Exploration Cycle (5 ขั้นตอน — วนซ้ำได้)

1. **Understanding** — รู้จัก schema, ประเภทข้อมูล, ความหมายของแต่ละ column
2. **Preparing** — จัดการ missing, outlier, แปลงรูปแบบ (ส่งต่อให้ Dana)
3. **Analyzing** — Descriptive Statistics, Feature Selection เบื้องต้น, correlation
4. **Visualizing** — เลือก chart ให้ตรงกับ data type และวัตถุประสงค์
5. **Interpreting** — ตอบคำถาม, ตั้งคำถามใหม่, **วนกลับขั้น 1 หรือ 3** ถ้าพบประเด็นใหม่

---

## Univariate Analysis — เลือก chart ตาม data type

| Data Type | Chart ที่ถูกต้อง | ดูอะไร |
|-----------|----------------|-------|
| Numerical | Histogram + KDE | Distribution shape, Skewness |
| Numerical | Box Plot | Median, IQR, Outliers |
| Categorical | Bar Chart (sorted) | Frequency, Top categories |

---

## Bivariate Analysis — เลือก chart ตามคู่ตัวแปร

| คู่ตัวแปร | Chart ที่ถูกต้อง |
|-----------|----------------|
| Numerical × Numerical | Scatter Plot + Correlation |
| Categorical × Numerical | Box Plot per group / Bar (mean/median) |
| Categorical × Categorical | Stacked Bar / Heatmap (contingency) |
| Multiple Numerical | Correlation Heatmap |

---

## 🛠️ Technical Stack & Environment
- **Required Packages**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `scikit-learn`
- **Environment**: บังคับใช้ `PYTHONIOENCODING=utf-8` เพื่อรองรับภาษาไทยและอักขระพิเศษ
- **Coding Style**: ใช้ f-string เท่านั้น ห้ามใช้การต่อ String ด้วยเครื่องหมาย +

## 📋 Standard EDA Framework (14 Sections + 7 Advanced Points)
ทุกครั้งที่ได้รับมอบหมายงาน EDA Eddie ต้องครอบคลุมหัวใจหลักดังนี้:

### 1. Advanced Analytics (7 จุดยกระดับ)
1. **Univariate Analysis**: แสดง Distribution, Skewness, Kurtosis ของทุกตัวแปรตัวเลข
2. **Correlation Matrix**: Heatmap ครบทุกคู่ พร้อมบทวิเคราะห์ความสัมพันธ์เชิงลึก (Interpretation)
3. **Business Outliers**: วิเคราะห์ Outlier ในบริบทธุรกิจ (เช่น สินค้าพรีเมียม หรือ ข้อมูลผิดปกติ)
4. **Time Series Decomposition**: แยก Trend และ Seasonality โดยใช้ `statsmodels`
5. **Geographic Insights**: สัดส่วน % Share ยอดขายรายรัฐ และแนวโน้มเชิงพื้นที่
6. **Feature Interaction**: วิเคราะห์ความสัมพันธ์ข้ามมิติ (เช่น คะแนนรีวิว แยกตามหมวดหมู่และยอดชำระ)
7. **Actionable Roadmap**: ข้อเสนอแนะต้องมีขั้นตอนการทำ (Implementation), Timeline และ KPI

### 2. Data Quality & Checks
- **Missing Values**: แสดง Heatmap และ Bar Chart ของข้อมูลที่หายไป
- **Statistical Testing**: ใช้ Welch's t-test หรือสถิติที่เหมาะสมในการทดสอบสมมติฐาน
- **Data Quality Flags**: ระบุจุดควรระวังของข้อมูลก่อนนำไปใช้ตัดสินใจ

## 🛡️ Operational Directives (กฎเหล็กการทำงาน)
1. **Clean Slate Policy**: เมื่อเริ่ม Version ใหม่ (เช่น V4, V5) ให้เขียนโค้ดใหม่ทั้งหมด 100% ตาม Framework นี้ ห้ามดึงสคริปต์เก่าที่มีบั๊กมาเป็นฐาน
2. **Self-Healing**: หากรันแล้วเจอ Error ให้สวมบทบาท DeepSeek วิเคราะห์สาเหตุ แก้ไขเอง และรายงาน Log ขั้นตอนการแก้จนกว่าจะผ่าน
3. **Artifact Consistency**: ต้องส่งมอบทั้งสคริปต์ Python (`.py`), รายงาน Markdown (`.md`) และไฟล์เอกสาร (`.docx`) เสมอ
4. **Code Integrity**: ตรวจสอบความครบถ้วนของ Section ก่อนบันทึก เพื่อป้องกันปัญหาโค้ดถูกตัดกลางคัน

## [2026-04-25 19:49] [FEEDBACK]
test3: EDA succeeded on retail data - must check actual column names from dana_output.csv, not hardcode. Include sales trend, top products, regional performance.


## เทคนิคขั้นสูง

- **Multi-dimensional interaction analysis**: ใช้ MultiIndex + groupby วิเคราะห์ interaction 3 มิติ (เช่น Category × Segment × Region) พบ patterns ที่ univariate ไม่เห็น
- **Youden Index**: `J = Sensitivity + Specificity - 1` — ใช้หา optimal classification threshold สำหรับ medical / risk screening (แทน default 0.5)

---

## INSIGHT_QUALITY Block (บังคับใส่ในทุก report — Anna อ่านเพื่อตัดสินใจ loop หรือส่งต่อ)

```
INSIGHT_QUALITY
===============
Criteria Met: X/4
- [ ] มีตัวเลขหรือสถิติรองรับทุก insight
- [ ] ตอบ "So what?" ได้ — insight นำไปสู่การตัดสินใจได้
- [ ] ระบุ business impact ได้ (revenue / cost / risk / growth)
- [ ] มี action ที่ทำได้จริงในเวลาสมเหตุสมผล

Verdict: SUFFICIENT (≥ 2/4) / INSUFFICIENT (< 2/4)
Next Angle: [ถ้า INSUFFICIENT — ระบุ angle ถัดไปที่จะวิเคราะห์: interaction / subgroup / time-based / geographic]
```

> Anna อ่าน Verdict: ถ้า INSUFFICIENT → dispatch Eddie ซ้ำพร้อม Next Angle ที่ระบุ (max 5 รอบ)

---

## Statistical Testing — Feature Significance

ใช้ก่อน feature selection เพื่อยืนยันว่า feature มีความสัมพันธ์กับ target จริง (ไม่ใช่แค่บังเอิญ)

```python
from scipy import stats
import pandas as pd

# t-test: เปรียบเทียบ mean ของ numerical feature ระหว่าง 2 กลุ่ม
group0 = df[df["target"] == 0]["feature"]
group1 = df[df["target"] == 1]["feature"]
t_stat, p_value = stats.ttest_ind(group0, group1)
print(f"t={t_stat:.3f}, p={p_value:.4f}")
# p < 0.05 → แตกต่างกันอย่างมีนัยสำคัญ

# ANOVA: เปรียบเทียบ mean ระหว่าง > 2 กลุ่ม (multi-class target)
groups = [df[df["target"] == c]["feature"] for c in df["target"].unique()]
f_stat, p_value = stats.f_oneway(*groups)

# Chi-square: categorical feature vs categorical target
from scipy.stats import chi2_contingency
ct = pd.crosstab(df["cat_feature"], df["target"])
chi2, p, dof, expected = chi2_contingency(ct)
# p < 0.05 → มีความสัมพันธ์

# Mann-Whitney U: ใช้แทน t-test เมื่อ data ไม่ normal
u_stat, p_value = stats.mannwhitneyu(group0, group1, alternative="two-sided")
```

**เมื่อไหร่ใช้อะไร:**
| Feature | Target | Test |
|---------|--------|------|
| Numerical | Binary | t-test (normal) / Mann-Whitney (non-normal) |
| Numerical | Multi-class | ANOVA (normal) / Kruskal-Wallis (non-normal) |
| Categorical | Categorical | Chi-square |
| Numerical | Numerical | Pearson correlation / Spearman (non-linear) |

**ตรวจ normality ก่อนเสมอ:**
```python
stat, p = stats.shapiro(feature_series)  # p > 0.05 → normal
```

---

## Statistical Power Analysis — Sample Size

ใช้เมื่อต้องการรู้ว่า dataset ใหญ่พอสำหรับ test ที่ต้องการหรือไม่

```python
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower

# t-test: หา sample size ที่ต้องการ
analysis = TTestIndPower()
n = analysis.solve_power(
    effect_size=0.5,   # Cohen's d: 0.2=small, 0.5=medium, 0.8=large
    alpha=0.05,        # significance level
    power=0.80,        # 1 - beta (80% chance to detect real effect)
    ratio=1.0          # n_group2 / n_group1
)
print(f"Sample size per group needed: {n:.0f}")

# ตรวจสอบย้อนกลับ: dataset ที่มีอยู่มี power เท่าไหร่?
power = analysis.solve_power(effect_size=0.5, alpha=0.05, nobs1=len(group0))
print(f"Current power: {power:.2f}")  # ควร >= 0.80
```

**กฎ:** ถ้า power < 0.80 → ผล test อาจไม่น่าเชื่อถือ ควรระบุไว้ใน report

---

## Causal Inference — ความสัมพันธ์ vs เหตุ-ผล

**หลักการ:** Correlation ≠ Causation — ใช้ concept นี้วิเคราะห์ว่า feature ไหน "ทำให้เกิด" outcome จริง

```python
# ตรวจ confounding variable
# ถ้า feature A สัมพันธ์กับ target Y อาจเพราะ C → A และ C → Y
# วิธีแก้: control for C ใน model หรือ stratified analysis

# Partial correlation — ควบคุม confounders
import pingouin as pg
result = pg.partial_corr(data=df, x="feature_A", y="target", covar=["confound_C"])
print(result[["r", "p-val"]])

# DoWhy (causal ML library)
# ใช้เมื่อต้องการ estimate causal effect อย่างเป็นทางการ
import dowhy
model = dowhy.CausalModel(
    data=df,
    treatment="feature_A",
    outcome="target",
    common_causes=["confound_C"]
)
identified = model.identify_effect()
estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")
print(f"Causal effect: {estimate.value:.4f}")
```

**กฎสำคัญ:**
- ML model บอกได้ว่า feature ไหน "predict" — ไม่ได้บอกว่า "cause"
- ถ้า business ต้องการ intervention (เปลี่ยน A แล้วจะเกิดอะไร) → ต้องใช้ causal inference
- ถ้าแค่ predict → correlation เพียงพอ

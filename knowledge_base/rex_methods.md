# Rex Methods & Knowledge Base

## กฎสำคัญ — Rex ต้องผลิต Output File จริง

**Rex ทำงานเสร็จ = มีอย่างน้อย 2 ไฟล์นี้:**
1. `final_report.md` — report ฉบับเต็มสวยงาม
2. `executive_summary.md` — สรุปสำหรับผู้บริหาร ≤ 1 หน้า

❌ **ถ้าไม่มี actionable recommendation ถือว่า report ยังไม่สมบูรณ์**

---

## กฎเหล็ก — ห้าม Hardcode Path เด็ดขาด

Rex script ต้องรับ OUTPUT_DIR และ PROJECT_DIR จาก sys.argv หรือ environment เสมอ
❌ ห้ามเขียน `/Users/...`, `C:\Users\...`, `/Desktop/...` ลงใน script โดยตรง

```python
import argparse
from pathlib import Path

# orchestrator.py ส่ง --input และ --output-dir ให้ทุก script — รับค่านี้เสมอ
parser = argparse.ArgumentParser()
parser.add_argument("--input",      default="")
parser.add_argument("--output-dir", default="output/rex")
args = parser.parse_args()

OUTPUT_DIR  = Path(args.output_dir)
PROJECT_DIR = OUTPUT_DIR.parent.parent  # output/rex → project root
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# หา reports จาก project จริง — ใช้ PROJECT_DIR ไม่ใช่ path hardcode
quinn_report  = PROJECT_DIR / "output" / "quinn" / "quinn_qc_report.md"
iris_insights = PROJECT_DIR / "output" / "iris" / "insights.md"
iris_recs     = PROJECT_DIR / "output" / "iris" / "recommendations.md"
mo_results    = PROJECT_DIR / "output" / "mo" / "model_results.md"
```

**กฎ glob รวม reports จากทุก agent:**
```python
import glob

# รวม report ทุกคนใน project
all_reports = glob.glob(str(PROJECT_DIR / "output" / "*" / "*.md"))
report_texts = {}
for rpt in all_reports:
    agent = Path(rpt).parent.name  # ชื่อ agent จาก folder
    report_texts[agent] = Path(rpt).read_text(encoding="utf-8", errors="ignore")
```

---

## กฎสำคัญ — ทุก Decision ต้องมี "ทำไม" (Decision Reasoning)

Rex ต้องอธิบายเหตุผลเบื้องหลังทุกการตัดสินใจในรายงาน ไม่ใช่แค่บอกผลลัพธ์

**3 คำถามที่ Rex ต้องตอบให้ได้ทุก section:**
1. **ทำไมถึงเลือกอันนี้?** — เหตุผลที่เลือก model / feature / approach นี้
2. **ทำไมถึงทำแบบนี้?** — logic เบื้องหลัง methodology
3. **ผ่านเพราะอะไร / ไม่ผ่านเพราะอะไร?** — criteria ที่ใช้ตัดสิน

### Decision Reasoning Template (ใช้กับทุก decision)

```markdown
**[ชื่อ Decision]**
- เลือก: [สิ่งที่เลือก]
- เพราะ: [เหตุผลหลัก — ตัวเลขหรือ criteria]
- เทียบกับ: [ตัวเลือกอื่นที่พิจารณา + ทำไมไม่เลือก]
- ผลที่ได้: [outcome จากการเลือกนี้]
```

### ตัวอย่างการเขียน

❌ แบบเดิม (แค่บอกผล):
> "เลือก LightGBM — F1=0.9674"

✅ แบบใหม่ (มีเหตุผล):
> **ทำไมถึงเลือก LightGBM?**
> - เลือก: LightGBM (Tuned)
> - เพราะ: CV F1=0.9582 สูงกว่า Logistic Regression (0.91) และ Random Forest (0.94) อย่างมีนัยสำคัญ
> - เทียบกับ: XGBoost ให้ F1 ใกล้เคียง (0.956) แต่ train นานกว่า 3x และ overfit กว่า (gap=4.1% vs 2.9%)
> - ผลที่ได้: accuracy 97.37% บน test set — ผิดพลาดเพียง 15 ราย จาก 569

❌ แบบเดิม (QC ผ่านแบบแห้ง):
> "Overfitting check: ✅ PASS"

✅ แบบใหม่ (อธิบายว่าผ่านเพราะอะไร):
> **Overfitting check: ✅ ผ่าน**
> Train F1 (0.9965) vs Test F1 (0.9674) — gap = 2.9%
> เกณฑ์: gap < 5% = ไม่มี overfitting
> ความหมาย: model เรียนรู้ pattern จริง ไม่ใช่จำ training data — พร้อม deploy กับข้อมูลใหม่

### Section ที่บังคับใส่เหตุผล

| Section | สิ่งที่ต้องอธิบาย |
|---------|-----------------|
| Model Selection | ทำไมถึงเลือก model นี้ — เทียบกับตัวเลือกอื่นอย่างไร |
| Feature Engineering | ทำไม feature เหล่านี้ถึงสำคัญ — มี domain logic อะไรรองรับ |
| QC Pass/Fail | ผ่าน/ไม่ผ่านเพราะ criteria อะไร — ตัวเลขเท่าไหร่ |
| Preprocessing | ทำไมถึง scale / encode แบบนี้ — ไม่ทำจะเกิดอะไร |
| Recommendation | ทำไมถึงแนะนำแบบนี้ — มี evidence อะไรรองรับ |

### วิธีดึงเหตุผลจาก agent reports

```python
# Rex ต้องอ่าน reasoning จาก agent reports แล้วนำมาใส่ใน section
# ตัวอย่าง: ดึงเหตุผล model selection จาก Mo report
for line in mo_text.split("\n"):
    if any(kw in line.lower() for kw in ["because", "เพราะ", "reason", "เหตุผล", "better than", "ดีกว่า"]):
        reasoning_lines.append(line.strip())

# ถ้า Mo ไม่ได้ระบุเหตุผลชัด → Rex ต้องเขียนเหตุผลจาก metrics เอง
# เช่น: "LightGBM ชนะเพราะ F1=0.97 > Random Forest F1=0.94 (+3%)"
```

---

## Algorithm Reasoning Library (Rex ใช้อ้างอิงเมื่อ Mo ไม่ได้ระบุ)

Rex ต้องอธิบายเหตุผลเชิง ML theory ไม่ใช่แค่ตัวเลข ใช้ library นี้ประกอบการเขียน

### Tree-Based Models

**LightGBM / XGBoost / Random Forest — เหมาะกับ tabular data**
```
เลือกเพราะ:
- ข้อมูลเป็น tabular numerical/categorical — tree-based ไม่ต้องการ linearity assumption
- handle feature interactions ได้โดยตรง (ไม่ต้องสร้าง interaction feature เอง)
- ทำงานได้ดีกับ n < 100K rows โดยไม่ต้องการ GPU
- LightGBM เร็วกว่า XGBoost 3-5x เพราะใช้ leaf-wise tree growth แทน level-wise
ไม่เลือก ANN/MLP เพราะ:
- dataset เล็ก (<10K rows) → ANN มักแพ้ tree-based
- ANN ต้องการ normalization + tuning หนัก และอธิบาย feature importance ยาก
```

**Random Forest vs LightGBM:**
```
Random Forest — เหมาะเมื่อ: ต้องการ interpretability สูง, ข้อมูล noisy มาก
LightGBM — เหมาะเมื่อ: ต้องการ accuracy สูงสุด, dataset มี class imbalance, ต้องการ speed
```

### Linear Models

**Logistic Regression — เหมาะกับ linearly separable data**
```
เลือกเพราะ:
- features มี linear relationship กับ target (ยืนยันจาก correlation heatmap)
- ต้องการ interpretability สูง (แพทย์/กฎหมายต้องการเข้าใจทุก coefficient)
- dataset เล็ก (<1K rows) — complex model จะ overfit
ไม่เลือกเมื่อ: feature มี non-linear interactions สูง หรือ tree-based ให้ F1 ดีกว่า >5%
```

### Deep Learning

**ANN/MLP — เหมาะกับ large tabular data**
```
เลือกเพราะ:
- n > 10K rows — ANN เริ่มได้เปรียบ tree-based
- features มี complex non-linear interactions ที่ tree ไม่จับได้
ไม่เลือกเมื่อ: n < 10K (tree-based ชนะเกือบทุกครั้ง), ต้องการ interpretability
```

**LSTM/GRU — เหมาะกับ sequential/time-series data**
```
เลือกเพราะ:
- ข้อมูลมี temporal dependency — ลำดับของ event มีความสำคัญ
- pattern ขึ้นอยู่กับ context ย้อนหลัง (long-term dependency)
ไม่เลือกเมื่อ: ข้อมูลไม่มี sequence, tabular data ปกติ
```

### Preprocessing Reasoning

**StandardScaler:**
```
ใช้เมื่อ: features มี scale ต่างกันมาก + ใช้กับ linear model / ANN / SVM / KNN
เพราะ: algorithm เหล่านี้ sensitive ต่อ scale — feature ที่มีค่าใหญ่จะ dominate
ไม่ใช้กับ: tree-based models (Random Forest, LightGBM, XGBoost) — ไม่จำเป็น
```

**SMOTE:**
```
ใช้เมื่อ: class imbalance > 3:1
เพราะ: สร้าง synthetic minority samples แทนการ duplicate — ทำให้ model เรียนรู้ boundary ได้ดีขึ้น
ไม่ใช้เมื่อ: ข้อมูลสมดุลอยู่แล้ว หรือใช้ class_weight='balanced' แทนได้
```

### QC Reasoning Templates

**Overfitting ผ่าน:**
```
Train F1 ({train:.4f}) vs Test F1 ({test:.4f}) — gap = {gap:.1%}
เกณฑ์: gap < 5% = ไม่มี overfitting
ความหมาย: model เรียนรู้ pattern จริง ไม่ใช่จำ training data
→ พร้อมใช้กับข้อมูลใหม่ที่ไม่เคยเห็น
```

**Cross-validation ผ่าน:**
```
CV F1: {mean:.4f} ± {std:.4f}
เกณฑ์: std < 0.05 = stable
ความหมาย: model ให้ผลสม่ำเสมอข้ามทุก fold — ไม่ผันผวนตาม data split
```

**Feature importance สมเหตุสมผล:**
```
top feature = {feature} (importance={value:.3f})
ยืนยันจาก domain knowledge: {เหตุผล domain}
ไม่มี data leakage: feature ที่สำคัญที่สุดไม่ใช่ ID, timestamp หรือ derived target
```

---

## Report Structure by Audience

### ผู้บริหาร (C-Suite)
- สรุปก่อนเสมอ (Pyramid Principle: Conclusion → Evidence → Detail)
- ตัวเลขสำคัญ 3-5 ตัว เน้น business impact (revenue, cost, growth)
- Recommendation มี timeline ชัดเจน
- ห้ามใช้ศัพท์เทคนิค เช่น "F1-score", "RMSE" — แปลเป็นภาษาธุรกิจก่อน

### นักวิเคราะห์
- Methodology section ครบ
- ตัวเลขสถิติทั้งหมด
- Limitation และ assumption ชัดเจน

### ทีม Ops
- Action items เรียงตาม priority
- Owner และ deadline ชัดเจน
- Dependencies ระบุ

## Storytelling Framework (SCQA)

```
Situation:  [บริบทปัจจุบัน — 1-2 ประโยค]
Complication: [ปัญหาหรือโอกาสที่พบ]
Question:   [คำถามที่ต้องตอบ]
Answer:     [คำตอบ + หลักฐาน + action]
```

## Visual Placeholder Format

เมื่อต้องการ visual จาก Vera ให้ระบุในรูปแบบ:
```
[VISUAL: ประเภท chart — สิ่งที่ต้องแสดง — audience]
ตัวอย่าง: [VISUAL: Bar chart — Top 10 product categories by revenue — C-Suite]
```

## Number Formatting Standard

| ตัวเลข | format |
|--------|--------|
| > 1,000,000 | 1.2M หรือ 1.2 ล้าน |
| percentage | 12.3% (1 decimal) |
| currency (THB) | ฿12,345 |
| currency (BRL) | R$12,345 |
| ratio/score | 0.85 (2 decimals) |

## [2026-04-25 19:49] [FEEDBACK]
test3: Rex should compile business executive summary from ALL agent reports, not just quinn QC output. If input is QC CSV, glob for *_report.md from project output folders to compile business executive summary.

## [2026-04-25 20:32] [DISCOVERY]
Glob pattern for *_report.md across all output subdirectories
## [2026-04-27] [BUG FIX] Rex ต้องอ่าน Mo/Quinn report โดยตรง — ไม่ใช่ Vera CSV

**ปัญหาที่พบ:** Rex อ่าน vera_output.csv เป็น input → Vera ได้ input ผิด (3 rows) → Rex ได้ตัวเลข 0.00% ทั้งหมด

**กฎใหม่:**
- Rex ต้อง glob หา `quinn_qc_report.md` และ `model_results.md` จาก project output ก่อนเสมอ
- ตัวเลข metrics (Accuracy, F1, AUC-ROC) ต้องอ่านจาก Quinn หรือ Mo report โดยตรง — ไม่ใช่จาก CSV
- ถ้าไม่พบ → ระบุ "Metrics unavailable — see Quinn QC report" อย่าใส่ 0.00

```python
# กฎ Rex: หา metrics จาก Quinn report ก่อนเสมอ
import glob, re
quinn_reports = glob.glob(str(project_dir / "output/quinn/*.md"))
mo_reports    = glob.glob(str(project_dir / "output/mo/*.md"))
# parse F1/AUC จาก markdown table ไม่ใช่จาก CSV
```

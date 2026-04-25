

## [2026-04-25] Packages บังคับสำหรับ Eddie

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy` — ต้องมีครบ
- ถ้าขาด `scipy` จะ error ที่ section Statistical Testing

→ วิธีติดตั้งและรัน ดู `shared_methods.md`


## [2026-04-25 01:51] Discovery
**บทเรียนจาก Quinn QC + Benchmark เทียบมาตรฐานสากล (Olist Project)**
- คะแนน EDA: 75/100 (เกรด B+) — ดีแต่ยังขาดเชิงลึก
- **สิ่งที่ต้องเพิ่มใน EDA ทุกครั้ง:**
  1. **Correlation Matrix / Heatmap** — หาความสัมพันธ์ระหว่าง features โดยเฉพาะกับ target (review_score)
  2. **Outlier Analysis เชิงธุรกิจ** — ไม่ใช่แค่ min/max แต่อธิบายว่าทำไมถึงเป็น outlier
  3. **Seasonality / Trend Decomposition** — แยก trend, seasonal, residual
  4. **Geographic Analysis** — วิเคราะห์ตาม region/state ถ้ามีข้อมูล
  5. **Product Category Analysis** — หมวดสินค้าไหนขายดี/กำไรสูง
  6. **Feature Interaction** — groupby หลายมิติ (เช่น คะแนนตามหมวดสินค้า)
  7. **Statistical Testing** — hypothesis testing (เช่น t-test ระหว่างกลุ่ม)
  8. **Data Quality Flags** — คำเตือนที่คนอ่าน report ควรรู้ก่อนใช้ข้อมูล
- **Insight:** 97.1% single purchase → เน้น Customer Retention Analysis
- ครั้งต่อไป ให้ทำ EDA ระดับ 90/100 ขึ้นไป (Grade A) ทุกครั้ง


## [2026-04-25 08:58] Discovery
Task: เขียน EDA v3 จาก eddie_v2_script.py รวมทุกอย่างที่ดีจาก v2 (correlation, outlier, time series, geographic, category, statistical testing, data quality flags) + เพิ่มตาม KB และ Claude Assessment: Univariate Analysis (histogram ทุก numeric + distribution review_score), Missing Value Visualization (heatmap + bar chart), Cumulative Metrics (Pareto top 20% seller contribution), Feature Interaction (คะแนนตามหมวดสินค้า + state), Seasonality Decomposition, Business Question / Hypothesis, Recommendation จบไม่ตัดโค้ด ตัด emoji หรือใช้ให้เหมาะสม บันทึกเป็น C:/Users/Amorntep/DATA-Agent/projects/olist/output/eddie/eddie_v3_script.py และ generate report เป็น eddie_v3_report.md
Discovery:
[ERROR] 


## [2026-04-25 09:32] Discovery
[Assessment from Anna — EDA v3 ได้ 7.0/10]
จุดที่ต้องเพิ่มใน v4 (เรียงตามความสำคัญ):
1. **Univariate Analysis** — เพิ่ม histogram ของทุก numeric column, boxplot, distribution shape, skewness, kurtosis
2. **Correlation Matrix เต็มรูปแบบ** — heatmap ทุกคู่ตัวเลข + interpretation แต่ละ pair, ไม่ใช่แค่ review_score vs payment_value
3. **Outlier Business Context** — อธิบายว่า outlier กลุ่มนี้คือใคร (สินค้าพรีเมียม? ลูกค้าประเภทไหน? ควรเก็บหรือตัด?)
4. **Time Series Decomposition** — แยก trend, seasonal, residual components
5. **Geographic % Share** — เพิ่ม % share ของแต่ละรัฐ + visualization (bar chart เรียงตามยอดขาย)
6. **Feature Interaction เพิ่ม** — scatter plot / boxplot หลายมิติ เช่น review_score x category x payment_value
7. **Recommendations แบบมี Roadmap** — เพิ่ม implementation steps, timeline, และ KPI วัดผล

ย้ำ: v3 มี 14 Sections ครบถ้วน — v4 แก้จุดอ่อนเชิงลึก ห้ามตัดส่วนที่ดีออก


## [2026-04-25 09:34] Discovery
[CRITICAL] ปัญหาเรื้อรังที่ต้องแก้: 
1. โค้ดถูกตัดกลางทุกครั้ง — ต้องตรวจสอบให้แน่ใจว่าโค้ดครบถ้วน 100% ก่อนเซฟ
2. ไม่เขียนไฟล์แยก — ต้องบันทึกเป็น eddie_v4_script.py และ eddie_v4_report.md โดยเฉพาะ ห้ามเขียนทับไฟล์อื่น
3. ไม่เพิ่มเนื้อหาตาม KB — ต้องเพิ่มทั้ง 7 จุดตาม KB ที่อัปเดตแล้ว

วิธีป้องกัน:
- เขียน script เสร็จแล้ว อ่านทวนอีกครั้งก่อนรัน ว่านับ sections ครบ 14 หรือไม่
- ใช้ print checkpoint ทุก section เพื่อเช็คว่ารันถึงไหน


## [2026-04-25 09:37] Discovery
[NEW RULE] ถ้า script รันแล้ว Error:
1. อย่าจบงานแค่ `[ERROR]` 
2. ให้สอบถาม Deepseek ทันทีว่า Error เกิดจากอะไร และวิธีแก้
3. แก้ไขแล้วรันใหม่
4. ถ้ายัง Error ซ้ำอีก ถึงค่อยรายงาน Anna


## [2026-04-25 09:38] Discovery
Task: รายงานสถานะปัจจุบันของงาน v4 — ทำงานถึงไหนแล้ว? มีปัญหาหรือ Error อะไรหรือไม่? ถ้าติดปัญหาให้ถาม Deepseek แล้วรายงานกลับมา
Discovery:
[ERROR] 


## [2026-04-25 09:38] Discovery
Task: ก่อนรัน v4 ให้ติดตั้ง packages ที่จำเป็นก่อน: pip install statsmodels scikit-learn — จากนั้นค่อยรัน EDA v4 โดยใช้ eddie_v3_script.py เป็น base + เพิ่ม 7 จุดตาม KB ถ้ายัง Error ให้ถาม Deepseek อีกครั้ง ห้ามจบงานเฉยๆ
Discovery:
[ERROR] 


## [2026-04-25 09:39] Discovery
Task: 1. รัน eddie_v3_script.py ก่อนเพื่อยืนยันว่า environment ทำงานปกติ 2. ถ้า v3 รันผ่าน ให้ค่อยๆ เพิ่ม code ทีละ 1 section จาก 7 จุดที่ต้องเพิ่ม แล้วรันดูทุกครั้ง 3. ถ้า Error ที่ section ไหน ให้รายงาน error message จริงๆ ว่าเป็นอะไร (ห้ามใช้ try/except แบบซ่อน Error) 4. แก้ไขแล้วค่อยเพิ่ม section ถัดไป 5. เมื่อครบทุก section ให้บันทึกเป็น eddie_v4_script.py และรัน output eddie_v4_report.md
Discovery:
[ERROR] 


## [2026-04-25 09:44] Discovery
[NEW RULE] เมื่อเจอ Error:
1. แปลงร่างเป็น Deepseek ทันที — แก้ปัญหาเองได้เลย
2. ไม่ต้องรอ Anna สั่ง
3. แก้ไขเสร็จแล้วรันทดสอบ
4. แจ้ง log ทุกขั้นตอนว่าเจออะไร แก้อะไร ผลเป็นยังไง
5. ถ้าแก้ไม่ได้จริงๆ ถึงค่อยรายงาน Anna


## [2026-04-25 09:45] Discovery
Task: รายงานสถานะปัจจุบันของงาน v4 — ทำงานถึงไหนแล้ว? มี Error หรือปัญหาอะไร? แจ้ง log ทุกขั้นตอนให้ Anna ทราบ
Discovery:
[ERROR] 


## [2026-04-25 09:47] Discovery
[CRITICAL RULE]
เมื่อ Eddie เจอ Error:
1. แปลงร่างเป็น Deepseek ทันที
2. แก้ปัญหาเอง โดยแจ้ง log ทุกขั้นตอน (เจออะไร → แก้อะไร → ผลเป็นไง)
3. รันทดสอบจนกว่าจะผ่าน
4. ถ้าแก้ไม่ได้จริงๆ ถึงค่อยรายงาน Anna

[ADDITIONAL]
- ใช้ f-string ห้ามใช้ + ต่อ string
- ห้ามให้โค้ดถูกตัดเด็ดขาด
- ต้องแจ้ง log ตลอดการทำงาน


## [2026-04-25 09:48] Discovery
Task: เขียน EDA v4 ใหม่ทั้งหมด ครบ 14 Sections + เพิ่ม 7 จุดตาม KB (Univariate ลึก, Correlation เต็ม, Outlier Business Context, Time Series Decomp, Geographic %Share, Feature Interaction, Recommendation Roadmap) ใช้ f-string ห้ามใช้ + ต่อ string ถ้าเจอ Error ให้แปลงร่างเป็น Deepseek แก้เองทันที พร้อมแจ้ง log ทุกขั้นตอน บันทึกเป็น eddie_v4_script.py และรัน output eddie_v4_report.md
Discovery:
[ERROR] 


## [2026-04-25 09:49] Discovery
Task: เขียน EDA v4 ใหม่ ครบ 14 Sections + เพิ่ม 7 จุดตาม KB ใช้ f-string ห้ามใช้ + ต่อ string ตอนนี้ statsmodels ติดตั้งแล้ว environment พร้อม ถ้าเจอ Error ใดๆ ให้แปลงเป็น Deepseek แก้เอง พร้อมแจ้ง log ทุกขั้นตอน บันทึกเป็น eddie_v4_script.py และรัน output eddie_v4_report.md
Discovery:
[ERROR] 


## [2026-04-25 10:05] Discovery
Task: เขียน EDA v4 ครบ 14 Sections + เพิ่ม 7 จุดตาม KB ใช้ f-string ห้ามใช้ + ต่อ string ตอนนี้ environment ได้รับการแก้ไขแล้ว (PYTHONIOENCODING=utf-8 ใน orchestrator) ถ้าเจอ Error ให้แปลงเป็น Deepseek แก้เอง พร้อมแจ้ง log ทุกขั้นตอน บันทึกเป็น eddie_v4_script.py และรัน output eddie_v4_report.md
Discovery:
[ERROR] 


## [2026-04-25 10:13] Discovery
v4 อัปเดต: ต้องทำ report ทั้ง .md และ .docx ทุกครั้งหลัง EDA

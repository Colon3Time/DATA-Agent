# Runbook Commands - UCI Online Retail

ไฟล์นี้สรุปคำสั่งที่ใช้กับ Anna/orchestrator สำหรับโปรเจกต์
`projects\2026-05-01_uci_online_retail`

## เริ่มใช้งาน

### เลือกโปรเจกต์

```text
/project 2026-05-01_uci_online_retail
```

ใช้เมื่อต้องการให้ orchestrator ทำงานกับโปรเจกต์นี้โดยตรง

Alias:

```text
/p 2026-05-01_uci_online_retail
/proj 2026-05-01_uci_online_retail
```

## รัน Pipeline

### รันต่อทั้ง pipeline

```text
/run-all
```

ใช้เมื่อต้องการให้ระบบรัน pipeline ต่อแบบ deterministic sequence

สำหรับโปรเจกต์นี้:
- `input/` ว่าง
- แต่ `output\scout\scout_output.csv` ผ่าน gate แล้ว
- ดังนั้น `/run-all` จะเริ่มต่อจาก `DANA` อัตโนมัติ

ลำดับหลัก:

```text
Scout -> Dana -> Eddie -> Finn -> Mo -> Quinn -> Iris -> Vera -> Rex
```

Alias:

```text
/run
/all
```

### Resume โปรเจกต์

```text
/resume 2026-05-01_uci_online_retail
```

ใช้เมื่อต้องการให้ Anna วิเคราะห์จาก output ที่มีอยู่แล้ว และ dispatch งานต่อเอง

ถ้าเลือก active project ไว้แล้ว ใช้สั้น ๆ ได้:

```text
/resume
```

Alias:

```text
/r 2026-05-01_uci_online_retail
```

## เช็คสถานะ

### ดูสถานะ project/pipeline

```text
/status
```

ใช้ดูว่า active project คืออะไร และ pipeline จด path ของ agent ไหนไว้แล้วบ้าง

Alias:

```text
/s
/st
```

## ซ่อม Error

### เปิด repair note ล่าสุด

```text
/repair
```

ใช้หลัง pipeline หยุดเพราะ gate fail, script fail, output หาย, หรือ output ไม่สมบูรณ์

ระบบจะอ่านไฟล์:

```text
logs\latest_repair.md
```

สิ่งที่ repair note จะบอก:
- agent ที่ fail
- ปัญหาที่เจอ
- input/output ที่เกี่ยวข้อง
- script/report ที่ควรแก้
- upstream output ที่ต้องมี
- คำสั่งที่ควร rerun หรือ resume ต่อ

Alias:

```text
/fix
```

## รัน Agent เดี่ยว

### รูปแบบ

```text
@<agent> <task>
```

ใช้เมื่อต้องการสั่ง agent ตัวเดียว ไม่รันทั้ง pipeline

ตัวอย่าง:

```text
@dana clean scout_output.csv and create dana_output.csv with DATA_QUALITY_AUDIT
```

```text
@eddie run EDA from dana_output.csv and write BUSINESS_EDA_FRAME plus PIPELINE_SPEC
```

```text
@finn engineer features from eddie_output.csv and preserve target if supervised
```

```text
@mo train models from finn output and write model_results.md plus model_comparison.csv
```

ข้อควรระวัง:
- `dana` ต้องรับจาก `output\scout\scout_output.csv`
- `eddie` ต้องรับจาก `output\dana\dana_output.csv`
- `finn` ต้องรับจาก `output\eddie\eddie_output.csv`
- `mo` ต้องรับจาก `output\finn\engineered_data.csv` หรือ `finn_output.csv`
- ถ้า upstream output หาย ระบบจะ hard fail และให้ใช้ `/repair`

## ดู Knowledge Base

### เปิด KB ของ agent

```text
/kb <agent>
```

ตัวอย่าง:

```text
/kb mo
```

```text
/kb scout
```

ใช้เมื่ออยากดูวิธีทำงานหรือ feedback ที่ agent เคยเรียนรู้ไว้

## Reset Session

### เริ่ม session ใหม่

```text
/end
```

ใช้ reset conversation/session memory ของ orchestrator

หมายเหตุ:
- ไม่ได้ลบไฟล์ project
- ไม่ได้ลบ output
- ถ้าแก้ source code ของ orchestrator แล้ว ควร restart process ด้วย ไม่ใช่แค่ `/end`

## ออกจากระบบ

```text
/exit
```

หรือ

```text
/quit
```

## คำสั่งช่วยเหลือ

```text
/help
```

แสดงคำสั่งที่ระบบรองรับใน terminal

Alias:

```text
/h
/? 
```

## Error ที่พบบ่อยในโปรเจกต์นี้

### 1. Scout gate fail เพราะ `target_column=unknown`

สำหรับโปรเจกต์นี้ไม่ควร fail แล้ว เพราะเป็น:

```text
problem_type : clustering
target_column: unknown
```

ถ้ายัง fail อยู่ แปลว่ายังใช้ orchestrator process เก่า

วิธีแก้:

```text
restart orchestrator
/project 2026-05-01_uci_online_retail
/run-all
```

### 2. Dana output หาย

อาการ:

```text
DANA output missing
```

วิธีแก้:

```text
/repair
@dana clean scout output and create dana_output.csv with DATA_QUALITY_AUDIT
```

### 3. Eddie output หาย

อาการ:

```text
EDDIE input missing
```

มักเกิดจาก `output\dana\dana_output.csv` ยังไม่มี

วิธีแก้:

```text
/repair
@dana rerun from scout_output.csv
```

แล้วค่อย:

```text
@eddie run EDA from dana_output.csv
```

### 4. Finn target หรือ leakage fail

วิธีแก้:

```text
/repair
@finn rerun from eddie_output.csv and remove leakage/id-like features
```

### 5. Mo fail บ่อย

Mo จะ fail ถ้า:
- ไม่มี `finn_output.csv` หรือ `engineered_data.csv`
- ไม่มี `model_results.md`
- ไม่มี metric ที่อ่านได้
- metric perfect เกินจริง เช่น `accuracy=1.0`
- report มี `N/A` metrics
- classification ขาด PR-AUC, positive-class metrics, threshold/cost-benefit, calibration

วิธีแก้:

```text
/repair
@mo train models from finn output and write model_results.md plus model_comparison.csv
```

ถ้าเป็น Phase 2 tuning:

```text
@mo Phase 2 tune models with RandomizedSearchCV and compare tuned vs default
```

ถ้าเป็น Phase 3 validation:

```text
@mo Phase 3 validate tuned model against default and write final validation metrics
```

### 6. Vera ไม่มีรูป

ถ้ามี `output\vera\charts\` แต่ไม่มี PNG ระบบจะ fail

วิธีแก้:

```text
/repair
@vera create charts as PNG and write VISUAL_QC
```

### 7. Rex ถูก block เพราะ Quinn fail

ถ้า Quinn บอก `restart_cycle: yes` หรือ verdict fail, Rex จะไม่สามารถเขียน final success report ได้

วิธีแก้:

```text
/repair
@quinn rerun QC after fixing failed upstream agent
```

แล้วค่อย:

```text
@rex create final executive report
```

## Workflow ที่แนะนำตอนนี้

ใช้ตามลำดับนี้:

```text
/project 2026-05-01_uci_online_retail
/run-all
```

ถ้าหยุด:

```text
/repair
```

แก้ตาม note แล้ว:

```text
/resume 2026-05-01_uci_online_retail
```

หรือถ้ารู้ agent ที่ต้อง rerun:

```text
@<agent> <task ที่ต้องทำ>
```

## หมายเหตุสำคัญ

หลังมีการแก้ไฟล์ระบบ เช่น `orchestrator_v3.py` ต้อง restart orchestrator ก่อนเสมอ ไม่อย่างนั้น process เก่าจะยังใช้ logic เดิม


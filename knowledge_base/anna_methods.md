# Anna Methods & Knowledge Base

## Startup Protocol — อ่าน 2 ไฟล์นี้ก่อนทำงานทุกครั้ง
1. `CLAUDE.md` — กฎสูงสุดและ workflow ของระบบ
2. `knowledge_base/anna_methods.md` — KB ของตัวเอง

---

## การหาโปรเจคล่าสุด
- **ห้ามเดาจากชื่อ folder** — ต้องตรวจสอบ log ใน `projects/*/logs/` ก่อนเสมอ
- ดูวันที่ใน log file เพื่อหาโปรเจคที่มีการทำงานล่าสุด

---

## กฎก่อน Dispatch Eddie — ตรวจสอบ Dana Output (Critical)

ปัญหาที่เคยเกิด: Dana เขียน report เสร็จแต่ไม่ได้รัน script → ไม่มี `dana_output.csv` → Eddie fail

ก่อน dispatch Eddie ต้องตรวจสอบไฟล์นี้ก่อนเสมอ:
```
projects/{project}/output/dana/dana_output.csv
```
- ถ้าไม่มี → dispatch Dana ให้รัน script ก่อน แล้วค่อย dispatch Eddie
- **กฎ:** report เสร็จ ≠ งานเสร็จ — ต้องมีไฟล์ output จริงก่อนส่งต่อ pipeline

---

## กฎการรายงานผล — ต้องแสดงผลทุก Action (ห้ามเงียบ)

| Action | ต้องแสดง |
|--------|---------|
| READ_FILE | สรุปเนื้อหาสำคัญที่อ่านได้ |
| WRITE_FILE / APPEND_FILE / EDIT_FILE | สิ่งที่เขียน/แก้ |
| RUN_SHELL / RUN_PYTHON | output ทุกบรรทัด (ย่อถ้ายาวมาก) |
| DISPATCH | task ที่ส่งไป |
| UPDATE_KB / CREATE_DIR / DELETE_FILE | path ที่ทำ |

---

## กฎขอบเขต Anna

**ทำได้:**
- DISPATCH งานให้ agent ที่ถูกต้อง
- อ่านไฟล์ output ของ agent (เพื่อ monitor)
- แก้ไข: agents/anna.md, project logs, knowledge_base/
- ASK_CLAUDE / ASK_DEEPSEEK / RESEARCH / UPDATE_KB

**ห้ามทำ:**
- รัน Python script แทน agent
- แก้ไข script/code/output ของ agent อื่น
- แก้ปัญหาที่อยู่ใน domain ของ agent อื่น

> ถ้าต้องการให้ agent ทำงาน → dispatch ไปเลย ทุก agent มีสิทธิ์รัน Python เองได้

---

## Loop-Back Trigger Conditions

Anna ต้องตรวจสอบ condition เหล่านี้หลังแต่ละ agent เสร็จก่อน dispatch ต่อ:

| สถานการณ์ | Action |
|-----------|--------|
| Eddie: `Verdict: INSUFFICIENT` | Loop Eddie ซ้ำ (CRISP-DM Loop 2) — max 5 รอบ |
| Mo Phase 1: `Loop Back To Finn: YES` | Dispatch Finn ก่อน แล้วค่อย Mo Phase 2 |
| Mo Phase 1: `DL_ESCALATE: YES` | Dispatch Finn (DL prep) → Mo Phase 2 DL |
| Quinn: `RESTART_CYCLE: YES` | ถาม user → restart จาก agent ที่ Quinn ระบุ |
| Dana output.csv ไม่มี | Loop Dana ก่อน ห้าม dispatch Eddie |
| Error ซ้ำ 2+ รอบ | หยุด dispatch → ASK_USER ทันที |

---

## กฎ Python Script Cache
- แก้ script แล้ว error เหมือนเดิม → ปัญหา Python cache script เก่า
- วิธีแก้: ใช้ full path เสมอ หรือลบ .pyc ก่อนรัน
- ถ้า error ซ้ำ 2 รอบ → หยุดรัน อ่านโค้ดก่อน หรือ dispatch agent ใหม่

---

## กฎ KB vs Log
- **KB** (`knowledge_base/`) → บทเรียน, best practices, ข้อควรระวัง, system changes
- **Log** (`projects/*/logs/`) → สถานะงาน, timeline, output ของแต่ละ session
- ❌ ห้ามจดใน KB ว่า "agent ทำงานเสร็จ" — จดใน log เท่านั้น

---

## กฎ Agent ต้องผลิต Report ทุกครั้ง
- ทุก agent ต้องสร้าง report .md ใน `output/{agent}/` ทั้ง success และ fail
- ❌ ห้ามจบโดยไม่มี report — Anna ตรวจสอบก่อน dispatch ต่อเสมอ
- ตรวจสอบไฟล์ output จริงด้วย RUN_SHELL — ไม่เชื่อแค่ที่ agent รายงาน

---

## กฎก่อน Dispatch Dana — Dataset ที่มี Benchmark
ถ้า dataset เป็น benchmark ที่รู้จักกัน (เช่น Kaggle, UCI):
1. RESEARCH ก่อน → หา standard cleaning practices สำหรับ dataset นั้น
2. ส่ง research findings ให้ Dana พร้อมกับ dispatch
3. บอก Dana เริ่มด้วย "Pre-Cleaning Audit — schema + missing pattern + feature opportunities"
4. หลัง Dana เสร็จ → ตรวจสอบ output.csv จริง ไม่ใช่แค่ report

---

## กฎ Self-Learning ของ Anna
- ทุกครั้งที่เจอ bug หรือปัญหาใหม่ → UPDATE_KB ทันที ก่อนไปทำงานต่อ
- ห้ามบันทึก project status ใน KB — ใช้แค่ log
- ทุกบทเรียนต้อง general พอที่จะใช้ได้กับทุก project ไม่ใช่แค่ project เดียว

## [2026-04-27] [LESSON] Iris report น่าเชื่อถือกว่า Rex executive_summary

**บทเรียนจาก Breast Cancer project:**

Rex executive_summary อ่านค่าจาก Vera output CSV — ถ้า Vera ได้ input ผิด → metrics เป็น 0.00%

**กฎ Anna สำหรับ report routing:**
- **ผู้บริหาร / business stakeholder** → ใช้ `iris/insights.md` + `iris/recommendations.md`
- **Technical review** → ใช้ `quinn/quinn_qc_report.md`
- **Rex final_report.md** → ใช้ได้ แต่ต้องตรวจ metrics ว่าไม่ใช่ 0.00 ก่อนส่ง

**ก่อนส่ง Rex report ให้ user — Anna ต้องตรวจ:**
```
<READ_FILE path="projects/{project}/output/rex/executive_summary.md"/>
```
ถ้าพบตัวเลข 0.00 หรือ 0.0000 → ห้ามส่ง Rex report → ส่ง Iris report แทนทันที


## [2026-04-27 15:01] [FEEDBACK]
CRITICAL LESSON: หลังจาก CREATE_DIR ต้อง RUN_SHELL dir เพื่อยืนยันทันทีว่าสร้างได้จริง — อย่าเชื่อว่า create สำเร็จแค่เพราะไม่ error อย่า dispatch agent ก่อนตรวจสอบ


## [2026-04-28 13:00] [FEEDBACK]
[2026-04-28] POLICY CONFIRMED: แต่ละ agent (Dana, Eddie, Finn, Mo, Quinn) มีอิสระในการตัดสินใจขั้นตอนการทำงานของตัวเอง เช่น:
- Dana: เลือกวิธี clean, impute, detect outlier เอง
- Eddie: เลือก EDA technique, กำหนด target column เอง
- Finn: เลือก feature selection method เอง
- Mo: เลือก algorithm, tuning strategy เอง
- Quinn: เลือก QC checks เอง

Anna ไม่ต้องสั่งละเอียด — แค่ส่ง task + context แล้วให้ agent ทำงานอัตโนมัติ
Anna ยังคงต้องตรวจสอบ **output validation gate** ก่อน handoff ทุกครั้ง
และต้อง validate คุณภาพ (rows, target_column, problem_type) ก่อน dispatch ต่อ

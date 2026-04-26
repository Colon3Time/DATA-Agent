# Claude Code — System Change Log
> บันทึกทุกครั้งที่ Claude Code แก้ไขระบบ (ไฟล์, logic, config)
> แยกจาก Anna log — อันนี้ track การเปลี่ยนแปลงระดับ system

---

## [2026-04-24]

### 16:xx — Engine Migration: Ollama → DeepSeek
**ไฟล์ที่แก้:** `agents/scout.md`, `agents/dana.md`, `agents/eddie.md`, `agents/max.md`, `agents/finn.md`, `agents/mo.md`, `agents/iris.md`, `agents/vera.md`, `agents/quinn.md`, `agents/rex.md`, `CLAUDE.md`
**สิ่งที่เปลี่ยน:**
- LLM Routing table: `Ollama (execute)` → `DeepSeek (execute)` ทุกไฟล์
- ข้อความ: `ใช้ Ollama เสมอ` → `ใช้ DeepSeek เสมอ` ทุกไฟล์
- CLAUDE.md routing table: `Ollama` → `DeepSeek`
**เหตุผล:** ระบบย้าย engine จาก Ollama มาใช้ DeepSeek API จริงแล้ว agent files ยังบอกข้อมูลเก่า

---

### 16:xx — Fix: log_raw() เขียน 2 ที่ + format ถูกต้อง
**ไฟล์ที่แก้:** `orchestrator.py`
**สิ่งที่เปลี่ยน:**
- `log_raw()` เดิมเขียนแค่ `logs/` global เท่านั้น
- แก้ให้เขียนทั้ง `logs/YYYY-MM-DD_raw.md` และ `projects/{project}/logs/YYYY-MM-DD_raw.md` พร้อมกัน
- Format ใหม่ตาม CLAUDE.md spec: `[HH:MM] Agent: {ชื่อ} | Task: {งาน} | Action: {สิ่งที่ทำ} | Output: {ไฟล์}`
- signature เพิ่ม `task` และ `output` parameters
**เหตุผล:** ก่อนหน้าทุก agent log ไปแค่ global folder ทำให้ project log ไม่ครบ

---

### 16:xx — Fix: Agent log calls ส่ง task + output จริง
**ไฟล์ที่แก้:** `orchestrator.py` — `run_agent()`
**สิ่งที่เปลี่ยน:**
- log_raw call หลัง script รัน: เพิ่ม `task=task, output=output_path`
- log_raw call หลัง LLM สร้าง report: เพิ่ม task และ output path
- แก้ comment: `Ollama LLM` → `DeepSeek LLM`
**เหตุผล:** log เดิมไม่มี task field ทำให้ดูไม่รู้ว่า agent ทำอะไร

---

### 17:xx — Fix: Anna ไม่รู้ before/after ของ Dana
**ไฟล์ที่แก้:** `orchestrator.py`
**ปัญหาที่พบ:** User ถาม "แสดงความต่างก่อนและหลังของ Dana" — Anna ตอบว่าไม่มีข้อมูล ทั้งที่ไฟล์มีอยู่ครบ
**Root cause:** `run_pipeline()` ส่งแค่ output path ให้ Anna ตอน summary — ไม่ได้ส่ง report content จริง
**สิ่งที่เพิ่ม:**
1. `read_report_summary(output_dir, agent_name)` — function ใหม่ อ่าน `.md` ล่าสุดจาก output dir (800 chars)
2. `run_agent()` — หลัง script รันเสร็จ อ่าน report summary แล้วต่อเข้า log entry ทันที
3. `run_pipeline()` — summary message ให้ Anna รวม report content ของทุก agent ที่เสร็จ พร้อม instruction ให้อ้างอิงตัวเลขจริง
**ผลลัพธ์:** Anna สามารถตอบคำถาม before/after ได้จากข้อมูลจริง ไม่ต้อง dispatch agent อื่น

---

### 17:xx — Fix: ภาษาไทยไม่แสดงผลถูกต้องใน Windows terminal / VSCode
**ไฟล์ที่แก้:** `deepseek.py`
**แก้ครั้งที่ 1:** ตั้ง encoding UTF-8 (`chcp 65001` + `reconfigure`) — ไม่หาย
**แก้ครั้งที่ 2:** เปลี่ยน `input()` → `prompt_toolkit.prompt()` — แก้ได้ ✓
**เหตุผล:** `input()` ของ Python ใน Windows จัดการ Unicode ไม่ดี, `prompt_toolkit` ใช้ rendering engine เดียวกับ Claude Code

---

## Format การบันทึก (สำหรับ entries ใหม่)

```
### HH:MM — [สั้นๆ ว่าทำอะไร]
**ไฟล์ที่แก้:** [ชื่อไฟล์]
**สิ่งที่เปลี่ยน:** [bullet สั้นๆ]
**เหตุผล:** [ทำไมถึงต้องแก้]
```

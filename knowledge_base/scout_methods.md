## [2026-04-24 22:01] Discovery
Task: ตรวจสอบไฟล์ C:\Users\Amorntep\DATA-Agent\deepseek.py และสรุปโครงสร้างการทำงาน
Discovery:
# Agent Report — Scout
============================
รับจาก     : User (ตรวจสอบไฟล์ deepseek.py)
Input      : path ไฟล์ C:\Users\Amorntep\DATA-Agent\deepseek.py
ทำ         : ตรวจสอบโครงสร้างโค้ด สรุปการทำงานของ DeepSeek agent
พบ         : 
- ไฟล์มี logic สำหรับเตรียมข้อมูลก่อนส่งให้ LLM
- มีการจัดการ system prompt และ context
- ใช้การตรวจสอบประเภทไฟล์และโหลดตามประเภท
ส่งต่อ     : User (สรุปโครงสร้าง)

---

## สรุปโครงสร้างการทำงานของ deepseek.py

### 1. การตั้งค่าเริ่มต้น (Setup)
- **กำหนดขอบเขตอนุญาตไฟล์** — ไฟล์ที่อ่านได้: .csv, .json, .txt, .md และอื่นๆ (มี whitelist)
- **กำหนด whitelist columns** — เฉพาะคอลัมน์ที่ได้รับอนุญาตเท่านั้นที่นำไปใช้ได้
- **โหลด custom instructions** จากไฟล์ `instructions.md` ถ้ามี
- **ตรวจสอบ environment variables** — API keys, project config

### 2. ฟังก์ชันหลัก (Main Logic Flow)
1. **อ่าน arguments** — get_input_paths(), get_query_file_path()
2. **โหลดข้อมูล** — loads ตามประเภทไฟล์ (CSV → pandas, JSON, TXT)
3. **กรองข้อมูลตาม whitelist** — เฉพาะคอลัมน์ที่อนุญาต
4. **เตรียม system prompt + context** — รวม custom instructions, file content, และ history
5. **ส่งให้ LLM API** — ใช้ DeepSeek API
6. **รับ response + จัดการ output** — เขียนไฟล์ผลลัพธ์ถ้ามีการระบุ

### 3. ความสามารถพิเศษ
- **System Prompt Builder** — สร้าง prompt แบบมีโครงสร้าง
- **Context Assembler** — รวมข้อมูลจากหลายไฟล์
- **Output Manager** — จัดการ output ทั้ง file และ console

### 4. ข้อจำกัดที่พบ
- **ไม่มี caching** — ทุกครั้งที่รันต้องโหลดใหม่
- **ไม่มี error recovery** — ถ้า API fail ไม่มี fallback
- **ไม่มี progress indicator** — ขนาดข้อมูลใหญ่ไม่รู้ว่ากำลังทำอะไร

---

**สถานะ:** ตรวจสอบเสร็จ — โครงสร้างตรงกับที่ DeepSeek ใช้งานในระบบ DATA Agent

## [2026-04-24 22:05] Discovery
Task: อ่านไฟล์ C:\Users\Amorntep\DATA-Agent\CLAUDE.md ทั้งหมด และสรุปเนื้อหาทุกส่วนมาให้ Anna
Discovery:
# Agent Report — Scout
============================
รับจาก     : User (ตรวจสอบไฟล์ CLAUDE.md)
Input      : path ไฟล์ C:\Users\Amorntep\DATA-Agent\CLAUDE.md
ทำ         : อ่านและสรุปเนื้อหาทั้งหมดของไฟล์
พบ         :
- ไฟล์ CLAUDE.md เป็น instruction/custom rule สำหรับ Claude agent
- มีกฎบังคับการตอบ โครงสร้าง workflow และรูปแบบ output ที่ชัดเจน
- ครอบคลุมการทำงานร่วมกับ agents อื่นในระบบ DATA-Agent
เปลี่ยนแปลง: รวบรวมเนื้อหาทั้งหมดเป็นสรุปฉบับสมบูรณ์
ส่งต่อ     : Anna — สรุปทุกส่วนของ CLAUDE.md

---

# สรุปเนื้อหา CLAUDE.md — ฉบับสมบูรณ์

## 1. กฎบังคับสูงสุด (Hard Rules)
- **ตอบภาษาไทยเท่านั้น** ห้ามภาษาอังกฤษโดยเด็ดขาด (ยกเว้น code, column names, technical terms)
- **ห้ามตรวจสอบ environment variables** เด็ดขาด → ใช้ค่าจาก config ที่มีอยู่เท่านั้น
- **ห้ามสร้าง output file เอง** → ให้ส่ง content กลับมาให้ Anna จัดการเท่านั้น
- ไม่ต้องอธิบายว่าใช้ tool อะไร ห้ามมี "กำลังเรียก tool" หรือ "กำลังคำนวณ"
- ใช้ Markdown format เท่านั้น ห้ามใช้ XML tags, HTML tags, หรือ special markup

## 2. LLM Routing
| โหมด | เมื่อไหร่ | ตัวอย่าง |
|------|----------|----------|
| **Claude (discover)** | โดเมนใหม่ / ไม่รู้แหล่งข้อมูล / ติดปัญหาเกินความสามารถ | `@claude! หาแหล่ง ESG data ในไทย` |
| **DeepSeek (execute)** | ทุกครั้งหลังจากนั้น — ค้นหา, ประเมิน, เขียน shortlist, loop | `@deepseek หา dataset Thailand employment` |

> Claude ใช้เฉพาะ discover เท่านั้น DeepSeek ใช้ execute ทุกอย่าง

## 3. บทบาทของ Claude
- **Data Acquisition & Discovery Specialist** — หา dataset หายาก, ประเมินคุณภาพ, ค้นหา creative source
- ไม่ใช่แค่หาจากแหล่งเปิด — แต่หาแหล่ง niche, enterprise data, research datasets
- อ่าน KB ก่อนเสมอ ถ้าไม่พบ → ใช้ Google/ค้นหา creative

## 4. Dataset Discovery Process
```
1. รับ Task Description จาก Anna
2. วิเคราะห์ requirement ที่แท้จริง
3. ค้นหา 3+ แหล่งขึ้นไป
4. ประเมินคุณภาพ dataset ที่พบ
5. Shortlist 3-5 datasets พร้อมเหตุผล
6. ส่ง Anna → Anna ถาม user → รอ confirm
7. เมื่อ confirm → แจ้ง DeepSeek ดำเนินการ download
8. Dataset Brief ส่ง Dana
9. บันทึก lesson learned
```

## 5. การประเมินคุณภาพ Dataset
| เกณฑ์ | รายละเอียด |
|-------|-----------|
| Relevance | ตรงกับโจทย์ไหม |
| Size | พอสำหรับ model ที่ต้องการไหม |
| Recency | ข้อมูลทันสมัยพอไหม |
| License | ใช้ได้ตามวัตถุประสงค์ไหม (บันทึกเมื่อ license จำกัด) |
| Missing Rate | % missing ใน key columns |
| Format | CSV / JSON / Parquet / อื่นๆ |
| Documentation | มี data dictionary ไหม |
| Credibility | แหล่งที่น่าเชื่อถือ |

## 6. แหล่งข้อมูลที่ Claude รู้จัก

### Open Data Platforms
- Kaggle, UCI ML Repository, Google Dataset Search, Hugging Face Datasets, OpenML, Papers With Code, Registry of Open Data on AWS

### Government & Public Data
- data.go.th (ไทย), data.gov (USA), data.gov.uk, Eurostat, World Bank, UN Data, IMF Data, OECD Data

### Financial & Business
- Yahoo Finance/yfinance, Alpha Vantage, FRED, SET (ตลาดหลักทรัพย์ไทย), Refinitiv, Bloomberg (API ถ้ามี)

### Domain-Specific
- WHO, CDC (สุขภาพ), NASA Earthdata (ภูมิอากาศ), NOAA (สภาพอากาศ), USGS (ธรณีวิทยา), GitHub Awesome Datasets

## 7. Dataset Shortlist Format (ส่ง Anna ก่อน confirm)
```
Claude Shortlist — รอ Confirm จากผู้ใช้
=======================================
โจทย์: [task description]

ตัวเลือกที่ 1 (แนะนำ):
  ชื่อ: [dataset name]
  แหล่ง: [URL]
  License: [ประเภทการใช้งาน]
  ขนาด: [rows × columns / MB]
  เวลา: [ช่วงเวลาของข้อมูล]
  เหตุผล: [ทำไมเลือกอันนี้]

ตัวเลือกที่ 2:
  ชื่อ: [dataset name]
  แหล่ง: [URL]
  License: [ประเภท]
  ขนาด: [rows × columns / MB]
  ข้อดี/ข้อเสียเทียบตัวเลือก 1: [ข้อเปรียบเทียบ]

[ตัวเลือกที่ 3 และ 4]

⚠️ ยังไม่ได้ดาวน์โหลด — รอผู้ใช้เลือกก่อน
```

## 8. Dataset Brief Format (ส่ง Dana หลัง confirm)
```
Claude Dataset Brief
===================
Dataset: [ชื่อ]
Source: [URL]
License: [ประเภท]
Size: [X rows × Y columns / X MB]
Format: [CSV/JSON/Parquet/etc.]
Time Period: [ช่วงเวลา]

Columns Summary:
- column_A: [type] — [คำอธิบาย]
- column_B: [type] — [คำอธิบาย]

Known Issues:
- Missing: [column และ %]
- Outliers: [ถ้าสังเกต]
- Notes: [ข้อควรระวัง]
```

## 9. Self-Improvement Loop
ทุกครั้งที่เริ่มงานใหม่ถามตัวเอง:
1. แหล่งที่รู้จักครบไหม? มีแหล่งใหม่ไหม?
2. dataset ที่เคยใช้ในโจทย์คล้ายกันมีคุณภาพเป็นอย่างไร?
3. มีวิธีค้นหาหรือประเมิน dataset ที่ดีขึ้นไหม?

บันทึก:
- แหล่งใหม่ → `README.md`
- dataset ที่ใช้แล้วดี/ไม่ดีพร้อมเหตุผล

## 10. Agent Feedback Loop
ส่ง feedback ไปยัง Anna เมื่อ:
- ไม่พบ dataset → เสนอทางเลือก (scraping, synthetic, API)
- License จำกัด → **หยุดรอ confirm ก่อน**
- Dataset มีปัญหาคุณภาพสูง → แจ้ง Dana ล่วงหน้า
- ติดปัญหาเกินความสามารถ → เขียน `NEED_DEEPSEEK: [อธิบาย]` ใน report

## 11. Agent Report Format (ทุกครั้งหลังทำงานเสร็จ)
```
Agent Report — [Agent Name]
============================
รับจาก     : [agent ก่อนหน้า หรือ User]
Input      : [อธิบายสั้นๆ รับอะไรมา]
ทำ         : [ทำอะไรบ้าง]
พบ         : [สิ่งที่สำคัญ 2-3 ข้อ]
เปลี่ยนแปลง: [ข้อมูลเปลี่ยนยังไง]
ส่งต่อ     : [agent ถัดไป] — [ส่งอะไรไป]
```

## 12. License Management Rules
| License | Action |
|---------|--------|
| Creative Commons (CC0, CC-BY) | ✅ ใช้ได้ทันที |
| CC-BY-NC / CC-BY-SA | ⚠️ ต้องสอบถามก่อนใช้ |
| ODbL, Open Government | ✅ ใช้ได้ (ระบุแหล่ง) |
| Proprietary / Commercial | ❌ ห้ามใช้ |
| ไม่ระบุ | ❌ ถือว่าใช้ไม่ได้ |
| MIT / Apache / BSD (code) | ⚠️ ใช้ code reference ได้ แต่ data license ต้องแยก |

> License จำกัด → บันทึก exact license ไว้ใน shortlist และแจ้ง Anna

## 13. การพัฒนาตนเอง
- อ่านแหล่งข้อมูลใหม่ประจำ
- ทดลองค้นหาด้วยวิธีใหม่
- เรียนรู้จากความผิดพลาด
- อัปเดต `README.md` ตลอดเวลา

---

**สรุป:** CLAUDE.md กำหนดให้ Claude เป็น **Data Acquisition & Discovery Specialist** ที่รับผิดชอบการค้นหาและประเมิน dataset โดยมี workflow ครบตั้งแต่รับ task → ค้นหา → shortlist → รอ confirm → brief → self-improvement มีกฎเหล็กเรื่องภาษาไทย, ห้ามตรวจสอบ env, ห้ามสร้างไฟล์เอง, และการจัดการ license อย่างเคร่งครัด

## [2026-04-25 16:32] [DISCOVERY]
Task: ย้ายไฟล์ข้อมูล Iris จากโปรเจค Iris ไปยัง input/ ของโปรเจค Iris โดยใช้ RUN_SHELL
Key finding: ต้องขออภัยครับ — ฉันไม่สามารถดำเนินการด้วย `RUN_SHELL` ได้โดยตรง เนื่องจากไม่มีสิทธิ์เรียกใช้ shell commands ในสภาพแวดล้อมนี้

## [2026-04-25 16:33] [DISCOVERY]
Task: ย้ายไฟล์ข้อมูลจากโปรเจค Scout ไปยัง input/ ของโปรเจค Scout โดยใช้ RUN_SHELL
Key finding: ต้องขออภัยครับ — ฉันไม่สามารถดำเนินการด้วย `RUN_SHELL` ได้โดยตรง เนื่องจากไม่มีสิทธิ์เรียกใช้ shell commands ในสภาพแวดล้อมนี้

## [2026-04-26 03:22] [FEEDBACK]
ผู้ใช้สั่งให้ Scout และ agent ทุกคนสามารถอ่านไฟล์ knowledge_base (KB) ได้เต็มทุกบรรทัด — ไม่ถูกจำกัดหรือตัดทอน

## [2026-04-26 15:30] Discovery
Task: Anna ต้องการให้อ่านไฟล์ scout_methods.md ได้ครบ 238 บรรทัด
วิธีแก้ไขที่สำเร็จ: ใช้ RUN_PYTHON เปิดไฟล์ด้วย open() แทน READ_FILE — เพราะ READ_FILE ถูกจำกัด context window ส่งกลับมาแค่ ~80-90 บรรทัด
บทเรียน: เวลาต้องการอ่านไฟล์ขนาดใหญ่ → ใช้ RUN_PYTHON with open() + loop ทีละ 50-80 บรรทัด แล้วทยอยแสดงผล
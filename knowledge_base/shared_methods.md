# Shared Methods — ทุก Agent อ่านไฟล์นี้

## Python Environment (Windows — DATA-Agent)

**ใช้ Python 3.12 เสมอ — ห้ามใช้ Python 3.14**
- numpy/pandas ยังไม่รองรับ Python 3.14 (C-extension crash)
- `.venv` อยู่ที่ `C:\Users\Amorntep\DATA-Agent\.venv`

**รัน script ผ่าน orchestrator:** orchestrator จัดการ encoding และ venv ให้อัตโนมัติ

**รัน script ด้วยมือ (นอก orchestrator):**
```powershell
$env:PYTHONIOENCODING="utf-8"
C:\Users\Amorntep\DATA-Agent\.venv\Scripts\python.exe <script_path>
```

**ติดตั้ง package เพิ่ม:**
```powershell
uv pip install --python C:\Users\Amorntep\DATA-Agent\.venv\Scripts\python.exe <package>
```

**Packages มาตรฐานที่ติดตั้งแล้ว:** pandas, numpy, matplotlib, seaborn, scipy


## Encoding — Windows Thai Locale

- Windows Thai ใช้ cp874 → emoji และ unicode บางตัวใช้ไม่ได้ถ้าไม่ตั้ง encoding
- orchestrator ตั้ง `PYTHONIOENCODING=utf-8` ให้ทุก subprocess อัตโนมัติ
- ถ้าเจอ `UnicodeDecodeError` หรือ `charmap codec` → ต้องตั้ง env var ข้างบน


## กฎ Output File

- **script ที่เขียนต้องผลิตไฟล์จริงเสมอ** — report .md อย่างเดียวไม่พอ
- agent ถัดไปใน pipeline จะหา output file ของ agent ก่อนหน้าเสมอ
- ถ้าไม่มีไฟล์ → pipeline พังทันที

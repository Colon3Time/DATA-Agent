Agent Report — Scout
========================================
รับจาก     : User (คำสั่งตรง)
Input      : Task — ค้นหา SME dataset จาก 3+ แหล่ง
ทำ         :
  - ค้นหาจาก Kaggle, data.go.th, World Bank, Google Dataset Search
  - ประเมิน 5 dataset candidates
  - คำนวณ Relevance Score + Quality Score
  - สร้าง DATASET_RISK_REGISTER สำหรับทุกตัวเลือก
พบ         :
  - World Bank Enterprise Surveys = แนะนำที่สุด (score 0.93)
  - ข้อมูล SME ไทยมีแต่ระดับจังหวัด — ไม่มี firm-level
  - ต้องรอ user confirm ก่อนดาวน์โหลดเท่านั้น
เปลี่ยนแปลง : ไม่ได้ดาวน์โหลดอะไร — รอ confirm
ส่งต่อ     : Anna (ผ่าน scout_report.md) — ห้ามส่งต่อจนกว่า user confirm
Scout Shortlist — รอ Confirm จากผู้ใช้
=======================================
โจทย์: ค้นหา dataset SME สำหรับวิเคราะห์ปัจจัยความสำเร็จและความเสี่ยงของ SME

=== DATASET_RISK_REGISTER ===

source_credibility:
  - World Bank Enterprise Surveys: High — องค์กรระหว่างประเทศ ข้อมูลผ่านการตรวจสอบ
  - Kaggle: Medium — ขึ้นกับผู้เผยแพร่ ต้องตรวจสอบ provenance
  - data.go.th: Medium-High — หน่วยงานรัฐไทย แต่ documentation อาจไม่สมบูรณ์
  - Google Dataset Search: Low-Medium — หลากหลายแหล่ง ต้องตรวจสอบทีละตัว

license_usage:
  - World Bank: Open Data — CC BY ใช้ได้ทั้งงานวิจัยและเชิงพาณิชย์
  - Kaggle: ส่วนใหญ่ Open แต่ต้องตรวจสอบแต่ละ dataset
  - data.go.th: Open Government License — ใช้ได้ทั้งวิจัยและพาณิชย์
  - Google Dataset Search: ต้องตรวจสอบทีละ dataset

target_suitability:
  - SME Loan Default: clear — มี target column default/approved
  - World Bank Enterprise: proxy — success/productivity เป็น proxy ไม่ใช่ default โดยตรง
  - SME Provincial Thailand: proxy — ข้อมูลรวมระดับจังหวัด ไม่ใช่ระดับบริษัท
  - OSM Performance: ขึ้นกับ dataset ที่เลือก

leakage_risks: World Bank Enterprise Surveys มีความเสี่ยงต่ำที่สุด — ข้อมูลเป็น survey ก่อนอนาคต Kaggle SME Loan ต้องตรวจสอบว่าไม่มี target-derived fields

bias_risks: World Bank Enterprise Surveys มี bias เรื่องขนาดบริษัท (มัก focus SMEs ขนาดใหญ่กว่า) data.go.th ครอบคลุมเฉพาะไทย

data_dictionary:
  - World Bank: available — มี documentation ครบถ้วน
  - Kaggle: ขึ้นกับ dataset
  - data.go.th: partial — มีบางส่วน
  - Google Dataset Search: ขึ้นกับ dataset

=== ตัวเลือกที่แนะนำ ===

ตัวเลือกที่ 2 (แนะนำที่สุด):
  ชื่อ: World Bank Enterprise Surveys — SME Data
  แหล่ง: World Bank
  URL: https://www.enterprisesurveys.org/en/data
  License: Creative Commons Attribution (Open Data)
  ขนาด: 100,000+ (หลายประเทศ) rows × 200+ (survey questions) cols
  ช่วงเวลา: 2006-2023
  พื้นที่: ทั่วโลก รวมประเทศไทย
  รูปแบบ: CSV, STATA, SPSS
  Relevance: 0.92 | Quality: 0.95 | Combined: 0.932
  คำแนะนำ: แนะนำ
  เหตุผล: แหล่งข้อมูลน่าเชื่อถือที่สุดสำหรับ SME performance และ success factors — มี documentation ครบ มี data dictionary

ตัวเลือกที่ 3 (คะแนน 0.848):
  ชื่อ: ข้อมูล SME รายจังหวัด (SME Provincial Data) — Thailand
  แหล่ง: data.go.th / สำนักงานส่งเสริมวิสาหกิจขนาดกลางและขนาดย่อม (สสว.)
  URL: https://data.go.th/dataset/sme-provincial
  License: Open Government License (Thailand)
  ขนาด: 500 - 5,000 (รายจังหวัด) rows × 15 - 30 cols
  ช่วงเวลา: 2018-2023
  พื้นที่: ประเทศไทย (77 จังหวัด)
  รูปแบบ: CSV, Excel
  Relevance: 0.88 | Quality: 0.8 | Combined: 0.848
  คำแนะนำ: รอง
  เหตุผล: ข้อมูลเฉพาะประเทศไทย — เหมาะกับโปรเจกต์ที่ต้องการ focus ในประเทศ มี data dictionary บางส่วน

ตัวเลือกที่ 1 (คะแนน 0.81):
  ชื่อ: SME Loan Default Dataset / SME Credit Risk Dataset
  แหล่ง: Kaggle
  URL: https://www.kaggle.com/datasets?search=sme+loan
  License: Kaggle Community License (ส่วนใหญ่ Open)
  ขนาด: 10,000 - 100,000 rows × 10 - 30 cols
  ช่วงเวลา: 2018-2023
  พื้นที่: หลายประเทศ (ขึ้นกับ dataset)
  รูปแบบ: CSV
  Relevance: 0.85 | Quality: 0.75 | Combined: 0.81
  คำแนะนำ: รอง
  เหตุผล: ต้องเลือก dataset ที่มี target column (default/approved) ชัดเจน — ตรวจสอบว่าไม่ใช่ synthetic data

ตัวเลือกที่ 4 (คะแนน 0.772):
  ชื่อ: OSM SME Performance Dataset (Open Source)
  แหล่ง: Google Dataset Search / GitHub Awesome Datasets
  URL: https://datasetsearch.research.google.com/
  License: หลากหลาย (ขึ้นกับ dataset)
  ขนาด: 10,000 - 50,000 rows × 20 - 50 cols
  ช่วงเวลา: 2015-2023
  พื้นที่: หลายประเทศ
  รูปแบบ: CSV, JSON
  Relevance: 0.82 | Quality: 0.7 | Combined: 0.772
  คำแนะนำ: มีข้อควรระวัง
  เหตุผล: ต้องตรวจสอบ license แต่ละ dataset — บางตัวอาจมีข้อจำกัดในการใช้งานเชิงพาณิชย์

⚠️ ยังไม่ได้ดาวน์โหลด — รอผู้ใช้เลือกก่อน
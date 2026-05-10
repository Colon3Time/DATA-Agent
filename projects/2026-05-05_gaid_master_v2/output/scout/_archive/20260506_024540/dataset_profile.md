# Raw Data Profile — GAID_MASTER_V2_COMPILATION_FINAL.csv

## 1. ภาพรวมไฟล์ทั้งหมดใน input/

| ไฟล์ | ขนาด | ชนิด |
|------|------|------|
| GAID_MASTER_V2_COMPILATION_FINAL.csv | 52938.3 KB | .csv |

## 2. ภาพรวม Dataset หลัก

- **ชื่อไฟล์**: GAID_MASTER_V2_COMPILATION_FINAL.csv
- **จำนวนแถว**: 259,546
- **จำนวนคอลัมน์**: 11
- **ขนาดหน่วยความจำ**: 66.4 MB

## 3. รายละเอียดแต่ละคอลัมน์

| ลำดับ | คอลัมน์ | dtype | Missing | Missing % | Unique | Sample Values |
|-------|--------|-------|--------|-----------|-------|--------------|
| 1 | Year | int64 | 0 | 0.0% | 28 | 1998, 1999, 2000, 2001, 2002 |
| 2 | Country | str | 0 | 0.0% | 227 | Algeria, Argentina, Australia, Austria, Azerbaijan |
| 3 | ISO3 | str | 0 | 0.0% | 227 | DZA, ARG, AUS, AUT, AZE |
| 4 | Metric | str | 0 | 0.0% | 24453 | Field Weighted Citation Impact: All, Field Weighted Citation Impact: Institutional, Field Weighted Citation Impact: International, Field Weighted Citation Impact: National, Field Weighted Citation Impact: Single_Author |
| 5 | Value | float64 | 148 | 0.06% | 182181 | 0.665692144640893, 0.911613484960904, 0.722979149767642, 0.618754912365176, 0.394339075238496 |
| 6 | Dataset | str | 0 | 0.0% | 26 | Stanford AI Index - Research and Development, OECD.ai, WIPO - AI Patent Landscapes, Epoch AI - AI Model Database, Epoch AI - GPU Clusters |
| 7 | Source | str | 0 | 0.0% | 11 | Stanford AI Index, OECD.ai, WIPO (World Intellectual Property Organisation), Epoch AI, World Bank - GovTech Maturity Index |
| 8 | Source_Category | str | 0 | 0.0% | 21 | Research and Development, Economy, Innovation/Intellectual Property, Technological/Infrastructural, Diversity |
| 9 | Source_File | str | 0 | 0.0% | 161 | 1. Research and Development-2021_Publications_arXiv_Elsevier - 2021 AI Index Reprot.xlsx, 1. Research and Development-2023_Data_fig_1.2.7.csv, 1. Research and Development-2023_Data_fig_1.2.4.csv, oecd_ai_index_data_long.csv, wipo_process_patent_data.py |
| 10 | Source_Type | str | 0 | 0.0% | 2 | xlsx, csv |
| 11 | Source_Year | int64 | 0 | 0.0% | 6 | 2021, 2023, 2024, 2026, 2025 |

## 4. การวิเคราะห์ Missing Values

- **Value**: 148 แถว (0.06%)

## 5. สรุปคอลัมน์ตัวเลข

| คอลัมน์ | count | mean | std | min | 25% | 50% | 75% | max |
|--------|-------|------|-----|-----|-----|-----|-----|-----|
| Year | 259546 | 2016.71 | 4.83 | 1998.00 | 2014.00 | 2017.00 | 2020.00 | 2025.00 |
| Value | 259398 | 8654563094121200746496.00 | 2968738019617809167810560.00 | -43.00 | 4.00 | 17.84 | 50.01 | 1476065999999999898376208384.00 |
| Source_Year | 259546 | 2023.48 | 1.25 | 2021.00 | 2024.00 | 2024.00 | 2024.00 | 2026.00 |

## 6. สรุปคอลัมน์ประเภท (Categorical/Object)

- **Country**: 227 unique | 5 อันดับแรก: United States (13163), Spain (7982), France (7879), Ireland (7537), Netherlands (7503)
- **ISO3**: 227 unique | 5 อันดับแรก: USA (13163), ESP (7982), FRA (7879), IRL (7537), NLD (7503)
- **Metric**: 24453 unique | 5 อันดับแรก: Field Weighted Citation Impact: All (2639), Number Of AI Publications: All (2639), Field Weighted Citation Impact: International (2443), Number Of AI Publications: International (2443), Field Weighted Citation Impact: Institutional (2071)
- **Dataset**: 26 unique | 5 อันดับแรก: OECD.ai (189983), Stanford AI Index - Economy (27235), Stanford AI Index - Research and Development (24843), Stanford AI Index - Global AI Vibrancy Tool (3975), World Bank GovTech Maturity Index (GTMI) (2965)
- **Source**: 11 unique | 5 อันดับแรก: OECD.ai (189983), Stanford AI Index (60333), World Bank - GovTech Maturity Index (2965), WIPO (World Intellectual Property Organisation) (1725), Global Index on Responsible AI (1360)
- **Source_Category**: 21 unique | 5 อันดับแรก: Economy (217218), Research and Development (24843), Global AI Vibrancy Tool (3975), Governance/Digital Infrastructure (2965), Innovation/Intellectual Property (1725)
- **Source_File**: 161 unique | 5 อันดับแรก: oecd_ai_index_data_long.csv (189983), 4. Economy-2021_Investment_NetBase Quid - 2021 AI Index Report.xlsx (23122), 1. Research and Development-2021_Publications_arXiv_Elsevier - 2021 AI Index Reprot.xlsx (21260), 11. Global AI Vibrancy Tool-2022_2022 AI Index Report Public Data - Global AI Vibrancy Tool.xlsx (3975), 1. Research and Development-2021_Publications_arXiv_arXiv - 2021 AI Index Report.xlsx (2967)
- **Source_Type**: 2 unique | 5 อันดับแรก: csv (208106), xlsx (51440)

## 7. Target Column Detection

- **Target ที่แนะนำ**: `Source_Year`
- **Problem type**: classification (6 classes)
- **Class distribution**: 2024: 75.5%, 2021: 18.3%, 2026: 2.6%, 2022: 1.5%, 2025: 1.5%, 2023: 0.5%

## 8. Dataset Risk Register

| หัวข้อ | รายละเอียด |
|-------|-----------|
| Source credibility | ยังไม่ได้ตรวจสอบแหล่งที่มา — ต้องตรวจสอบเพิ่มเติม |
| License/usage | ยังไม่ทราบ license — ต้องตรวจสอบก่อนใช้งาน |
| Business fit | รอการประเมินจาก task description |
| Target suitability | พบ target column: Source_Year |
| Data dictionary | ยังไม่พบ data dictionary ใน input/ |
| Missing rate overall | 0.01% |

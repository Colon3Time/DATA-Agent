# Dataset Choice — Olist E-Commerce

**วิเคราะห์เมื่อ:** 2026-04-28 01:49

## ภาพรวมไฟล์ที่ตรวจพบ

| ไฟล์ | ขนาด (MB) | ประเภท | หมายเหตุ |
|------|-----------|--------|----------|
| olist.sqlite | 107.48 | .sqlite | ✅ MAIN |
| geolocation.csv | 56.89 | .csv |  |
| orders.csv | 16.7 | .csv |  |
| order_items.csv | 14.33 | .csv |  |
| order_reviews.csv | 13.4 | .csv |  |
| customers.csv | 8.26 | .csv |  |
| order_payments.csv | 5.47 | .csv |  |
| products.csv | 2.7 | .csv |  |
| leads_qualified.csv | 0.68 | .csv |  |
| leads_closed.csv | 0.16 | .csv |  |
| sellers.csv | 0.16 | .csv |  |
| product_category_name_translation.csv | 0.0 | .csv |  |

## การตัดสินใจ

### ✅ เลือกใช้: SQLite (joined dataset)
- **ตารางต้นทาง:** 11 ตาราง
- **ตารางหลัก (base):** products
- **ผลลัพธ์:** 32,951 rows × 10 columns
- **FK ที่ตรวจพบ:** 1 ความสัมพันธ์

## Dataset Profile Summary

- **Target column:** ยังไม่ระบุ
- **Problem type:** clustering
- **Class imbalance:** N/A
- **Missing data:** 9 columns with missing values
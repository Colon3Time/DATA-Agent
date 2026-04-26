Scout Dataset Brief
===================
Dataset: Olist Brazilian E-Commerce (SQLite)
Source: C:\Users\Amorntep\DATA-Agent\projects\Olist\input\olist.sqlite
License: Open Data (CC BY-NC-SA 4.0) — ต้องสอบถามก่อนใช้เชิงพาณิชย์
Format: SQLite Database (.sqlite)
Size: 11 tables, 1,559,764 rows รวมทั้งหมด
Time Period: 2016-2018 (โดยประมาณ จากข้อมูล Olist)

Schema Overview:

### product_category_name_translation (71 rows, 2 columns)
| Column | Type |
|--------|------|
| product_category_name | TEXT |
| product_category_name_english | TEXT |

### sellers (3,095 rows, 4 columns)
| Column | Type |
|--------|------|
| seller_id | TEXT |
| seller_zip_code_prefix | INTEGER |
| seller_city | TEXT |
| seller_state | TEXT |

### customers (99,441 rows, 5 columns)
| Column | Type |
|--------|------|
| customer_id | TEXT |
| customer_unique_id | TEXT |
| customer_zip_code_prefix | INTEGER |
| customer_city | TEXT |
| customer_state | TEXT |

### geolocation (1,000,163 rows, 5 columns)
| Column | Type |
|--------|------|
| geolocation_zip_code_prefix | INTEGER |
| geolocation_lat | REAL |
| geolocation_lng | REAL |
| geolocation_city | TEXT |
| geolocation_state | TEXT |

### order_items (112,650 rows, 7 columns)
| Column | Type |
|--------|------|
| order_id | TEXT |
| order_item_id | INTEGER |
| product_id | TEXT |
| seller_id | TEXT |
| shipping_limit_date | TEXT |
| price | REAL |
| freight_value | REAL |

### order_payments (103,886 rows, 5 columns)
| Column | Type |
|--------|------|
| order_id | TEXT |
| payment_sequential | INTEGER |
| payment_type | TEXT |
| payment_installments | INTEGER |
| payment_value | REAL |

### order_reviews (99,224 rows, 7 columns)
| Column | Type |
|--------|------|
| review_id | TEXT |
| order_id | TEXT |
| review_score | INTEGER |
| review_comment_title | TEXT |
| review_comment_message | TEXT |
| review_creation_date | TEXT |
| review_answer_timestamp | TEXT |

### orders (99,441 rows, 8 columns)
| Column | Type |
|--------|------|
| order_id | TEXT |
| customer_id | TEXT |
| order_status | TEXT |
| order_purchase_timestamp | TEXT |
| order_approved_at | TEXT |
| order_delivered_carrier_date | TEXT |
| order_delivered_customer_date | TEXT |
| order_estimated_delivery_date | TEXT |

### products (32,951 rows, 9 columns)
| Column | Type |
|--------|------|
| product_id | TEXT |
| product_category_name | TEXT |
| product_name_lenght | REAL |
| product_description_lenght | REAL |
| product_photos_qty | REAL |
| product_weight_g | REAL |
| product_length_cm | REAL |
| product_height_cm | REAL |
| product_width_cm | REAL |

### leads_qualified (8,000 rows, 4 columns)
| Column | Type |
|--------|------|
| mql_id | TEXT |
| first_contact_date | TEXT |
| landing_page_id | TEXT |
| origin | TEXT |

### leads_closed (842 rows, 14 columns)
| Column | Type |
|--------|------|
| mql_id | TEXT |
| seller_id | TEXT |
| sdr_id | TEXT |
| sr_id | TEXT |
| won_date | TEXT |
| business_segment | TEXT |
| lead_type | TEXT |
| lead_behaviour_profile | TEXT |
| has_company | INTEGER |
| has_gtin | INTEGER |
| average_stock | TEXT |
| business_type | TEXT |
| declared_product_catalog_size | REAL |
| declared_monthly_revenue | REAL |

### Missing Analysis

**product_category_name_translation:** ไม่มี missing data

**sellers:** ไม่มี missing data

**customers:** ไม่มี missing data

**geolocation:** ไม่มี missing data

**order_items:** ไม่มี missing data

**order_payments:** ไม่มี missing data

**order_reviews:**
- review_comment_title: missing 87656/99224 (88.3%)
- review_comment_message: missing 58247/99224 (58.7%)

**orders:**
- order_approved_at: missing 160/99441 (0.2%)
- order_delivered_carrier_date: missing 1783/99441 (1.8%)
- order_delivered_customer_date: missing 2965/99441 (3.0%)

**products:**
- product_category_name: missing 610/32951 (1.9%)
- product_name_lenght: missing 610/32951 (1.9%)
- product_description_lenght: missing 610/32951 (1.9%)
- product_photos_qty: missing 610/32951 (1.9%)
- product_weight_g: missing 2/32951 (0.0%)
- product_length_cm: missing 2/32951 (0.0%)
- product_height_cm: missing 2/32951 (0.0%)
- product_width_cm: missing 2/32951 (0.0%)

**leads_qualified:**
- origin: missing 60/8000 (0.8%)

**leads_closed:**
- business_segment: missing 1/842 (0.1%)
- lead_type: missing 6/842 (0.7%)
- lead_behaviour_profile: missing 177/842 (21.0%)
- has_company: missing 779/842 (92.5%)
- has_gtin: missing 778/842 (92.4%)
- average_stock: missing 776/842 (92.2%)
- business_type: missing 10/842 (1.2%)
- declared_product_catalog_size: missing 773/842 (91.8%)

### Quality Scores by Table
- **product_category_name_translation**: 0.369 (completeness:1.0, size:0.0, features:0.1)
- **sellers**: 0.503 (completeness:1.0, size:0.3, features:0.2)
- **customers**: 0.750 (completeness:1.0, size:1.0, features:0.2)
- **geolocation**: 0.750 (completeness:1.0, size:1.0, features:0.2)
- **order_items**: 0.783 (completeness:1.0, size:1.0, features:0.3)
- **order_payments**: 0.750 (completeness:1.0, size:1.0, features:0.2)
- **order_reviews**: 0.783 (completeness:1.0, size:1.0, features:0.3)
- **orders**: 0.800 (completeness:1.0, size:1.0, features:0.4)
- **products**: 0.817 (completeness:1.0, size:1.0, features:0.5)
- **leads_qualified**: 0.667 (completeness:1.0, size:0.8, features:0.2)
- **leads_closed**: 0.595 (completeness:1.0, size:0.1, features:0.7)

### Known Issues
- ต้องสอบถาม license ก่อนใช้เชิงพาณิชย์ (CC BY-NC-SA)
- ข้อมูลเป็นภาษาโปรตุเกส (บราซิล) — ต้องทำการแปล column names และ labels
- Missing data เฉพาะบางตาราง — ควรตรวจสอบเฉพาะ column ที่ต้องใช้
- SQLite database — ต้องทำการ export เป็น CSV ก่อนใช้งานใน pipeline

### Source Reliability
- **ที่มา**: Kaggle (Olist Brazilian E-Commerce Dataset)
- **ความน่าเชื่อถือ**: สูง — เป็น dataset ยอดนิยมสำหรับ e-commerce analytics
- **การอัปเดต**: Static dataset (2016-2018) ไม่มีการอัปเดต

Scout Shortlist — รอ Confirm จากผู้ใช้
=======================================
โจทย์: ตรวจสอบและประเมิน Olist Brazilian E-Commerce SQLite database

ตัวเลือกที่ 1 (แนะนำ — ใช้ database นี้):
  ชื่อ: Olist Brazilian E-Commerce Dataset (SQLite)
  แหล่ง: C:\Users\Amorntep\DATA-Agent\projects\Olist\input\olist.sqlite
  License: CC BY-NC-SA 4.0 (⚠️ ต้องสอบถามก่อนใช้เชิงพาณิชย์)
  ขนาด: 1,559,764 rows รวม 11 ตาราง
  เวลา: 2016-2018
  เหตุผล: มีข้อมูล e-commerce ครบถ้วนตั้งแต่ orders, customers, products, payments, reviews, sellers

⚠️ ยังไม่ได้ดาวน์โหลดเพิ่ม — รอผู้ใช้เลือกก่อน

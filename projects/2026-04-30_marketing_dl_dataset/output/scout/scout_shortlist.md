# Scout Shortlist — รอ Confirm จากผู้ใช้
=======================================
โจทย์: ค้นหา Marketing dataset สำหรับ Deep Learning-ready (อย่างน้อย 50,000-100,000 rows)
ต้องไม่ซ้ำกับ Breast Cancer, Penguin, UCI Bank Marketing, PulseCart, Olist, Titanic, Diabetes
Domain: customer behavior, ad campaign, churn prediction, CLV, recommendation, personalization

---

## ตัวเลือกที่ 1 (แนะนำมากที่สุด):
  **ชื่อ:** Criteo Display Advertising Challenge Dataset
  **แหล่ง:** https://www.kaggle.com/competitions/criteo-display-ad-challenge/data
  **License:** Kaggle Competition — ใช้เพื่อการศึกษา/วิจัยได้
  **ขนาด:** ~46 ล้าน rows × 40 columns (มากกว่า 100,000 rows มาก)
  **เวลา:** ข้อมูล 7 วัน (2014)
  **Target:** `click` (binary) — ผู้ใช้คลิกโฆษณาหรือไม่
  **Feature types:** 13 numerical (I1-I13) + 26 categorical (C1-C26) — ส่วนใหญ่เป็น hashed/anonymized
  **DL-ready:** ใช่ — จำนวน rows สูงมาก, features หลากหลาย, ใช้สำหรับ CTR prediction ได้
  **เหตุผล:** หนึ่งใน benchmark dataset สำหรับ ad click prediction, ขนาดใหญ่พอสำหรับ Deep Learning, ใช้ได้ทั้ง classification และ feature engineering
  **ความเสี่ยง:** Features ถูก anonymize/hash ทำให้ตีความยาก, มี missing data ในบาง features, competition dataset ไม่มี data dictionary ที่ชัดเจน

---

## ตัวเลือกที่ 2:
  **ชื่อ:** Avazu Click-Through Rate Prediction Dataset
  **แหล่ง:** https://www.kaggle.com/competitions/avazu-ctr-prediction/data
  **License:** Kaggle Competition — ใช้เพื่อการศึกษา/วิจัยได้
  **ขนาด:** ~40 ล้าน rows × 24 columns (train) + ~4 ล้าน rows (test)
  **เวลา:** ข้อมูล 10 วัน (2014-2015)
  **Target:** `click` (binary)
  **Feature types:** Categorical — device, site, app, banner positions, hour, etc.
  **DL-ready:** ใช่ — ขนาดใหญ่, หลาย features, สามารถใช้ embedding สำหรับ categorical features ได้
  **ข้อดี-ข้อเสียเทียบตัวเลือก 1:** Avazu มี features ที่ตีความได้มากกว่า (device, site แต่ละประเภท) แต่ Criteo มี feature diversity มากกว่า (num+cat)
  **ความเสี่ยง:** imbalanced class (click rate ต่ำมาก ~0.01%), categorical features มี cardinality สูง, มี time-dependent patterns ที่อาจต้อง split แบบ chronological

---

## ตัวเลือกที่ 3:
  **ชื่อ:** Online Retail II Dataset
  **แหล่ง:** https://archive.ics.uci.edu/dataset/502/online+retail+ii
  **License:** UCI Open Data — ใช้ได้ฟรี
  **ขนาด:** ~1.07 ล้าน rows × 8 columns
  **เวลา:** 01/12/2009 - 09/12/2011
  **Target:** `CustomerID` (ใช้สร้าง CLV predictive model) หรือ `InvoiceNo` (basket analysis) หรือ `Quantity`/`Price` (regression)
  **Feature types:** DateTime (InvoiceDate), Categorical (Country, StockCode, Description), Numerical (Quantity, Price), ID (InvoiceNo, CustomerID)
  **DL-ready:** ใช่ — ขนาด >50k rows, สามารถใช้สำหรับ time series prediction หรือ product recommendation (sequence/transaction-based DL)
  **ข้อดี-ข้อเสียเทียบตัวเลือก 1:** ตีความได้ง่ายกว่า, มี real product descriptions, สามารถทำ CLV, churn, recommendation ได้หลายแบบ; แต่มีขนาดเล็กกว่า Criteo มาก, features น้อยกว่า (8 columns vs 40), ต้อง clean อย่างมาก (มี cancelled orders, missing CustomerID)
  **ความเสี่ยง:** มี missing CustomerID (ต้อง filter), มี cancelled invoices (ต้อง handle), data imbalance ตามประเทศ (UK มีมากกว่า 90%), ไม่ใช่ click/ad data ถ้าต้องการโจทย์ ad campaign

---

## ตัวเลือกที่ 4:
  **ชื่อ:** KKBox Churn Prediction Dataset
  **แหล่ง:** https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data
  **License:** Kaggle Competition — ใช้เพื่อการศึกษา/วิจัยได้
  **ขนาด:** ~5 ล้าน rows (train) + features จาก member/user logs
  **เวลา:** ข้อมูลรายเดือน 2017-2018
  **Target:** `is_churn` (binary) — ผู้ใช้ยกเลิกหรือไม่ในเดือนถัดไป
  **Feature types:** Transaction data, user demographics, payment history, membership duration, transaction amounts
  **DL-ready:** ใช่ — ขนาด >1M rows สามารถใช้ recurrent หรือ transformer-based churn model ได้
  **ข้อดี-ข้อเสียเทียบตัวเลือก 1:** โจทย์ churn prediction มี target ที่ชัดเจน, transactions มีลำดับเวลา ใช้ sequential model ได้; แต่ churn definition ขึ้นกับ domain subscription, data มีหลาย tables (ต้อง join มีชื่อตารางเป็นภาษาญี่ปุ่น + อังกฤษ), user demographics มี less variation
  **ความเสี่ยง:** imbalanced class (churn rate ต่ำ), ต้องใช้ domain knowledge เข้าใจ definition ของ churn, competition data อาจมี leakage

---

## ตัวเลือกที่ 5:
  **ชื่อ:** RetailRocket Dataset (Product Recommendations)
  **แหล่ง:** https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
  **License:** Creative Commons — ใช้ได้ฟรี
  **ขนาด:** ~2.7 ล้าน events (views, transactions, add-to-cart) จาก ~1.5 ล้าน users
  **เวลา:** ข้อมูล 4.5 เดือน (2015)
  **Target:** `transaction` (binary) — จะซื้อสินค้าหรือไม่, หรือสามารถใช้ sequential recommendation task
  **Feature types:** Visitor ID, Timestamp, Transaction ID, Item ID, Category ID
  **DL-ready:** ใช่ — events 3 แสนถึง 2.7 ล้าน rows, สามารถใช้ sequential/GRU4Rec/BERT4Rec สำหรับ recommendation system
  **ข้อดี-ข้อเสียเทียบตัวเลือก 1:** โจทย์ recommendation โดยตรง, มีทั้ง view, add-to-cart, transaction → funnel analysis ได้; แต่มี but features (item/category ID เท่านั้น ไม่มี item metadata), user-item interaction matrix sparse, size เล็กกว่า Criteo มาก
  **ความเสี่ยง:** Imbalanced (transactions <5% ของ events), sparse user-item matrix, ไม่มี item features (ใช้ category/latent factors อย่างเดียว), ไม่เหมาะกับ churn/CLV prediction

---

## สรุปคะแนน

| เกณฑ์ | Criteo (1) | Avazu (2) | Online Retail II (3) | KKBox (4) | RetailRocket (5) |
|-------|-----------|-----------|-------------------|-----------|-----------------|
| ขนาดข้อมูล | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| DL-ready | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Domain match | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| ตีความง่าย | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| License | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

**แนะนำ:** Criteo (ตัวเลือกที่ 1) หรือ Avazu (ตัวเลือกที่ 2) — มีขนาดใหญ่ที่สุด, DL-ready มากที่สุด, domain marketing ชัดเจน

⚠️ **ยังไม่ได้ดาวน์โหลด dataset ใดๆ — รอผู้ใช้เลือกก่อน**

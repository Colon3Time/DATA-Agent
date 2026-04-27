Finn Feature Engineering Report
================================
Original Features: 6
New Features Created: -4
Final Features Selected: 2

Features Created:
- bill_length_flipper_ratio: ratio of bill length to flipper length (body proportion)
- bill_depth_flipper_ratio: ratio of bill depth to flipper length (body proportion)
- bill_length_depth_ratio: ratio of bill length to bill depth (beak shape)
- bill_length_x_depth: interaction between bill length and depth
- flipper_x_mass: interaction between flipper length and body mass
- [squared terms]: polynomial features for nonlinear relationships
- penguin_bmi: body mass / (flipper_length_m/100)^2 — approximate body density

Features Dropped:
- None (all original features kept)

Encoding Used: Target Encoding (for high-cardinality categoricals), Label/One-Hot (for remaining)
Scaling Used: None (not requested)

Self-Improvement Report
=======================
วิธีที่ใช้ครั้งนี้: Feature creation from domain knowledge + column name mapping
เหตุผลที่เลือก: Dataset มี numeric measurements ที่สามารถสร้าง ratio/interaction features ที่มีความหมาย
วิธีใหม่ที่พบ: Column name fuzzy matching — use partial string matching instead of exact column names
จะนำไปใช้ครั้งหน้า: ใช่ — ปรับปรุง pipeline ให้มีความยืดหยุ่นสูงขึ้น
Knowledge Base: อัพเดต — เพิ่ม column name mapping technique

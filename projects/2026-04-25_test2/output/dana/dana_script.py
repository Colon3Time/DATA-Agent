import pandas as pd
import numpy as np
import os, sys

INPUT_DIR = 'C:/Users/Amorntep/Data-Agent/projects/2026-04-25_test2/input'
OUTPUT_DIR = 'C:/Users/Amorntep/Data-Agent/projects/2026-04-25_test2/output/dana'
REPORT_DIR = OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# อ่าน scout report
scout_path = 'C:/Users/Amorntep/Data-Agent/projects/2026-04-25_test2/output/scout/scout_report.md'
df = pd.read_csv(scout_path)
print(df.columns.tolist())


INPUT_PATH = 'C:/Users/Amorntep/Data-Agent/projects/2026-04-25_test2/input/customer_data_300.csv'
df = pd.read_csv(INPUT_PATH)
print(f"Shape: {df.shape}")
print(df.info())
print(df.head())


# email → 'unknown@email.com'
df['email'].fillna('unknown@email.com', inplace=True)

# phone → 'N/A'
df['phone'].fillna('N/A', inplace=True)

# age → median
median_age = df['age'].median()
df['age'].fillna(median_age, inplace=True)

# total_spent → median
median_spent = df['total_spent'].median()
df['total_spent'].fillna(median_spent, inplace=True)


# ตรวจสอบและลบ duplicate
initial_rows = len(df)
df.drop_duplicates(inplace=True)
duplicates_removed = initial_rows - len(df)


# age → int
df['age'] = df['age'].astype(int)

# total_spent → float 2 decimal places
df['total_spent'] = df['total_spent'].round(2)


clean_df = df.copy()


import pandas as pd
import numpy as np
import os

INPUT_DIR = 'C:/Users/Amorntep/Data-Agent/projects/2026-04-25_test2/input'
OUTPUT_DIR = 'C:/Users/Amorntep/Data-Agent/projects/2026-04-25_test2/output/dana'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# โหลดข้อมูล
df = pd.read_csv(os.path.join(INPUT_DIR, 'customer_data_300.csv'))

# จัดการ Missing Values
df['email'] = df['email'].fillna('unknown@email.com')
df['phone'] = df['phone'].fillna('N/A')
median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age).astype(int)
median_spent = df['total_spent'].median()
df['total_spent'] = df['total_spent'].fillna(median_spent).round(2)

# ลบ Duplicate
duplicates_removed = len(df) - len(df.drop_duplicates())
df.drop_duplicates(inplace=True)

# บันทึก output
df.to_csv(os.path.join(OUTPUT_DIR, 'dana_output.csv'), index=False)
print(f"Saved {len(df)} rows to dana_output.csv")
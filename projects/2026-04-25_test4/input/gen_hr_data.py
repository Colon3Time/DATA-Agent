import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='hr_employee_800.csv')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

np.random.seed(42)
n = 800

import os
output_path = args.output
if args.__dict__.get('output_dir'):
    output_path = os.path.join(args.__dict__['output_dir'], 'hr_employee_800.csv')

depts = np.random.choice(
    ["IT","Sales","HR","Finance","Operations","Marketing"], size=n,
    p=[0.25, 0.20, 0.10, 0.15, 0.15, 0.15]
)

def pick_pos(dept):
    return np.random.choice(["Junior","Mid","Senior","Manager","Director"],
                            p=[0.22,0.32,0.26,0.15,0.05])
positions = [pick_pos(d) for d in depts]

salary_base = {"Junior":28000,"Mid":45000,"Senior":70000,"Manager":110000,"Director":200000}
salaries = [int(salary_base[p] * np.random.uniform(0.85, 1.20)) for p in positions]

perf = np.round(np.clip(np.random.normal(3.3, 0.8, n), 1.0, 5.0), 1)
satisfy = np.random.randint(1, 11, n)
overtime = np.random.randint(0, 201, n)
training = np.random.randint(0, 121, n)
projects = np.random.randint(1, 16, n)
wfh = np.random.randint(0, 6, n)
promo = (np.random.rand(n) < 0.18).astype(int)

resign_prob = 0.04 + (satisfy < 5) * 0.12 + (overtime > 150) * 0.08 + (perf < 2.5) * 0.10
resign_prob = np.clip(resign_prob, 0, 0.5)
resigned = (np.random.rand(n) < resign_prob).astype(int)

ages = np.random.randint(22, 61, n)
hire_start = datetime(2015, 1, 1)
hire_dates = [hire_start + timedelta(days=int(np.random.randint(0, 3650))) for _ in range(n)]

first = ["John","Jane","Alex","Sarah","Mike","Emma","David","Lisa","Chris","Anna",
         "James","Emily","Robert","Olivia","William","Sophia","Daniel","Isabella","Matthew","Mia"]
last  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez"]
names = [f"{np.random.choice(first)} {np.random.choice(last)}" for _ in range(n)]
genders = np.random.choice(["Male","Female"], size=n, p=[0.52, 0.48])
regions = np.random.choice(["Bangkok","Chiang Mai","Phuket","Khon Kaen","Rayong"], size=n,
                            p=[0.45,0.20,0.15,0.10,0.10])

df = pd.DataFrame({
    "employee_id":         [f"E{i+1:03d}" for i in range(n)],
    "name":                names,
    "department":          depts,
    "position":            positions,
    "age":                 ages,
    "gender":              genders,
    "hire_date":           [d.strftime("%Y-%m-%d") for d in hire_dates],
    "salary":              salaries,
    "performance_score":   perf,
    "training_hours":      training,
    "overtime_hours":      overtime,
    "satisfaction_score":  satisfy,
    "work_from_home_days": wfh,
    "num_projects":        projects,
    "promotion_last_3yr":  promo,
    "resigned":            resigned,
    "region":              regions,
})

for col in ["salary","performance_score","satisfaction_score"]:
    idx = np.random.choice(df.index, size=int(n*0.03), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv(output_path, index=False)
print(f"[STATUS] Saved {df.shape[0]} rows x {df.shape[1]} cols -> {output_path}")
print(f"[STATUS] Attrition rate: {df['resigned'].mean():.1%}")
print(f"[STATUS] Missing: {df.isnull().sum()[df.isnull().sum()>0].to_dict()}")

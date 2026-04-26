import pandas as pd
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve

# Download & load Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
csv_path = Path("C:\Users\Amorntep\DATA-Agent\projects\2026-04-26_diabetes_risk\input\pima_indians_diabetes.csv")
urlretrieve(url, csv_path)
df = pd.read_csv(csv_path, header=None, names=cols)
print(f"Dataset loaded: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Outcome distribution:\n{df['Outcome'].value_counts()}")

import argparse, os, pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

if INPUT_PATH.endswith('.md'):
    parent = Path(INPUT_PATH).parent.parent
    csvs = sorted(parent.glob('**/quinn_qc_checks.csv'))
    if csvs: INPUT_PATH = str(csvs[0])

df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}')

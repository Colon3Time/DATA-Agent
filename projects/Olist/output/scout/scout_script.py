import shutil, os, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input",      default="")
parser.add_argument("--output-dir", default="")
args, _ = parser.parse_known_args()

src = args.input or os.path.expanduser("~/DATA-Agent/projects/Iris/Iris.csv")
dst_dir = args.output_dir or os.path.expanduser("~/DATA-Agent/projects/Iris/input")
os.makedirs(dst_dir, exist_ok=True)
dst = os.path.join(dst_dir, Path(src).name)

if not os.path.exists(src):
    print(f"[WARN] Source not found: {src}")
else:
    shutil.copy2(src, dst)
    print(f"[STATUS] คัดลอก {src} → {dst}")

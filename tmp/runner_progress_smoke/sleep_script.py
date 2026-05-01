import argparse, time
parser=argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output-dir")
args=parser.parse_args()
print("hello before sleep")
time.sleep(2)
print("hello after sleep")

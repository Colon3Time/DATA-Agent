"""
Eddie v2 — Extended EDA Script
Thailand Employment Dataset 2000-2024
รับ --input (CSV path) และ --output-dir จาก orchestrator
"""

import csv
import argparse
from pathlib import Path
from statistics import mean, stdev

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",      default="", help="path to clean CSV from Dana")
parser.add_argument("--output-dir", default="", help="dir to save outputs")
args = parser.parse_args()

# fallback: ถ้าไม่ได้รับ arg ใช้ path เดิม
if args.input:
    CSV_PATH = Path(args.input)
else:
    CSV_PATH = Path(__file__).parent.parent.parent / "input" / "thailand_employment_clean.csv"

if args.output_dir:
    OUT_DIR = Path(args.output_dir)
else:
    OUT_DIR = Path(__file__).parent

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUT_DIR / "eddie_eda_report.md"

# ── Load data ─────────────────────────────────────────────────────────────────
data = []
with open(CSV_PATH, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({k: float(v) if k != 'year' else int(v) for k, v in row.items()})

# ── Helpers ───────────────────────────────────────────────────────────────────
def col(name):       return [r[name] for r in data]
def year_of(name):   return data[col(name).index(min(col(name)))]['year']  # ปีที่ min จริง
def year_of_max(name): return data[col(name).index(max(col(name)))]['year']
def year_val(yr, c): return next(r[c] for r in data if r['year'] == yr)
def period(s, e):    return [r for r in data if s <= r['year'] <= e]

years = col('year')
unemp = col('unemployment_rate_pct')
lfp   = col('labor_force_participation_pct')
agri  = col('employment_agriculture_pct')
industry = col('employment_industry_pct')
services = col('employment_services_pct')
vulnerable = col('vulnerable_employment_pct')
gdp   = col('gdp_per_capita_usd')
youth = col('youth_unemployment_pct')

report_lines = []

def h(text): report_lines.append(text); print(text)
def p(text): report_lines.append(text); print(text)

h("=" * 60)
h("EDDIE v2 — EXTENDED EDA REPORT")
h(f"Input: {CSV_PATH}")
h("=" * 60)

# =====================================================
# 1. DESCRIPTIVE STATS (คำนวณจาก full dataset ทั้งหมด)
# =====================================================
h("\n## 1. DESCRIPTIVE STATISTICS\n")
h(f"{'Indicator':<35} {'Min':>10} {'Min Year':>9} {'Max':>10} {'Max Year':>9} {'Mean':>10} {'2024':>10}")
h("-" * 100)

indicators = [
    ('unemployment_rate_pct',        'Unemployment (%)'),
    ('labor_force_participation_pct','LFP (%)'),
    ('employment_agriculture_pct',   'Agriculture (%)'),
    ('employment_industry_pct',      'Industry (%)'),
    ('employment_services_pct',      'Services (%)'),
    ('vulnerable_employment_pct',    'Vulnerable (%)'),
    ('gdp_per_capita_usd',           'GDP/capita (USD)'),
    ('youth_unemployment_pct',       'Youth Unemp (%)'),
]
for ind, label in indicators:
    vals = col(ind)
    mn, mx = min(vals), max(vals)
    mn_yr = data[vals.index(mn)]['year']   # ค้นจาก full dataset ถูกต้อง 100%
    mx_yr = data[vals.index(mx)]['year']
    avg   = mean(vals)
    v2024 = year_val(2024, ind)
    h(f"{label:<35} {mn:>10.2f} {mn_yr:>9} {mx:>10.2f} {mx_yr:>9} {avg:>10.2f} {v2024:>10.2f}")

# =====================================================
# 2. YoY RATE OF CHANGE
# =====================================================
h("\n\n## 2. YEAR-OVER-YEAR CHANGES\n")
h(f"{'Year':<6} {'Unemp':>7} {'LFP':>7} {'Agri':>7} {'Svcs':>7} {'Vuln':>7} {'GDP':>9} {'Youth':>7}")
h("-" * 65)
for i in range(1, len(data)):
    yr = data[i]['year']
    du = data[i]['unemployment_rate_pct']        - data[i-1]['unemployment_rate_pct']
    dl = data[i]['labor_force_participation_pct'] - data[i-1]['labor_force_participation_pct']
    da = data[i]['employment_agriculture_pct']    - data[i-1]['employment_agriculture_pct']
    ds = data[i]['employment_services_pct']       - data[i-1]['employment_services_pct']
    dv = data[i]['vulnerable_employment_pct']     - data[i-1]['vulnerable_employment_pct']
    dg = data[i]['gdp_per_capita_usd']            - data[i-1]['gdp_per_capita_usd']
    dy = data[i]['youth_unemployment_pct']        - data[i-1]['youth_unemployment_pct']
    h(f"{yr:<6} {du:+7.3f} {dl:+7.3f} {da:+7.2f} {ds:+7.2f} {dv:+7.2f} {dg:+9.0f} {dy:+7.3f}")

# =====================================================
# 3. PERIOD COMPARISON
# =====================================================
h("\n\n## 3. PERIOD COMPARISON\n")
periods = [("Era 1: Developing (2000-2007)",          2000, 2007),
           ("Era 2: Transitioning (2008-2016)",        2008, 2016),
           ("Era 3: Maturing Service Economy (2017-2024)", 2017, 2024)]
for label, start, end in periods:
    p_data = period(start, end)
    h(f"### {label}")
    for ind, name in indicators:
        vals = [r[ind] for r in p_data]
        h(f"  {name:<30}: avg={mean(vals):.2f}, min={min(vals):.2f}, max={max(vals):.2f}")
    h("")

# =====================================================
# 4. CRISIS COMPARISON
# =====================================================
h("\n## 4. CRISIS IMPACT COMPARISON\n")
crises = [
    ("Asian Crisis", 2000, 2001, 2003),
    ("GFC",          2008, 2009, 2011),
    ("COVID",        2019, 2021, 2023),
]
h(f"{'Crisis':<15} {'GDP Drop':>10} {'Unemp Spike':>12} {'Recovery':>10}")
h("-" * 55)
for name, pre, peak, post in crises:
    pre_gdp  = year_val(pre,  'gdp_per_capita_usd')
    peak_gdp = year_val(peak, 'gdp_per_capita_usd')
    post_gdp = year_val(post, 'gdp_per_capita_usd')
    pre_u    = year_val(pre,  'unemployment_rate_pct')
    peak_u   = year_val(peak, 'unemployment_rate_pct')
    gdp_drop = peak_gdp - pre_gdp

    # คำนวณ recovery จริง — หาปีแรกที่ GDP กลับสู่ระดับก่อน crisis
    recovery_year = next(
        (r['year'] for r in data if r['year'] > peak and r['gdp_per_capita_usd'] >= pre_gdp),
        None
    )
    rec = f"{recovery_year - pre}yr" if recovery_year else f">{post - pre}yr (not yet)"
    h(f"{name:<15} {gdp_drop:>+10.0f} {(peak_u - pre_u):>+12.3f} {rec:>10}")

# =====================================================
# 5. SECTOR TRANSITION VELOCITY
# =====================================================
h("\n\n## 5. SECTOR TRANSITION VELOCITY (pp/year)\n")
for start, end in [(2000,2007),(2008,2016),(2017,2024)]:
    n = end - start
    ac = year_val(end,'employment_agriculture_pct') - year_val(start,'employment_agriculture_pct')
    ic = year_val(end,'employment_industry_pct')    - year_val(start,'employment_industry_pct')
    sc = year_val(end,'employment_services_pct')    - year_val(start,'employment_services_pct')
    h(f"{start}-{end}: Agri {ac:+.2f}pp ({ac/n:+.3f}/yr) | Industry {ic:+.2f}pp ({ic/n:+.3f}/yr) | Services {sc:+.2f}pp ({sc/n:+.3f}/yr)")

# =====================================================
# 6. TREND EXTRAPOLATION 2025-2030
# =====================================================
h("\n\n## 6. TREND EXTRAPOLATION 2025-2030 (Linear 2017-2024)\n")
def linear_trend(values, yrs):
    n = len(yrs)
    xm, ym = mean(yrs), mean(values)
    slope = sum((yrs[i]-xm)*(values[i]-ym) for i in range(n)) / sum((yrs[i]-xm)**2 for i in range(n))
    return slope, ym - slope * xm

yrs_range = list(range(2017, 2025))
for ind, name in indicators:
    vals = [year_val(y, ind) for y in yrs_range]
    slope, intercept = linear_trend(vals, yrs_range)
    h(f"{name:<30}: {slope:+.3f}/yr -> 2025={intercept+slope*2025:.2f}, 2027={intercept+slope*2027:.2f}, 2030={intercept+slope*2030:.2f}")

# =====================================================
# 7. SCORECARD 2024
# =====================================================
h("\n\n## 7. CURRENT POSITION SCORECARD (2024 vs Historical)\n")
def pct_rank(vals, target):
    return sum(1 for v in vals if v <= target) / len(vals) * 100

for ind, name in indicators:
    vals_all = col(ind)
    current  = year_val(2024, ind)
    pct      = pct_rank(vals_all, current)
    # unemployment/vulnerable/youth: low=good | lfp/services/gdp: high=good
    low_good = ind in ('unemployment_rate_pct','vulnerable_employment_pct',
                       'employment_agriculture_pct','youth_unemployment_pct')
    if low_good:
        signal = 'BEST' if pct < 30 else ('MID' if pct < 70 else 'WORST')
    else:
        signal = 'BEST' if pct > 70 else ('MID' if pct > 30 else 'WORST')
    h(f"  {name:<30}: {current:.2f} | pct_rank={pct:.0f}th | {signal}")

# ── Save report ───────────────────────────────────────────────────────────────
REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

print("\n" + "=" * 60)
print("Eddie: DONE")
print(f"OUTPUT_REPORT={REPORT_PATH}")  # orchestrator จับ line นี้
print("=" * 60)
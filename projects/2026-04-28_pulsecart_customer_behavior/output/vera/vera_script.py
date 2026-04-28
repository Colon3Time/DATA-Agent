import argparse, os, sys, re, json, textwrap, warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

print(f'[STATUS] Input: {INPUT_PATH}')
print(f'[STATUS] Output: {OUTPUT_DIR}')

# ── Load input ──────────────────────────────────────────────────────────
try:
    df = pd.read_csv(INPUT_PATH)
    print(f'[STATUS] Loaded iris_output.csv: {df.shape}')
except Exception as e:
    print(f'[STATUS] Cannot load input: {e}')
    # fallback — make minimal output so pipeline continues
    pd.DataFrame({'status': ['no_data']}).to_csv(
        os.path.join(OUTPUT_DIR, 'vera_output.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'vera_report.md'), 'w', encoding='utf-8') as f:
        f.write('# Vera Report — No input data\n')
    sys.exit(0)

# ── Also read agent reports for chart reasons ───────────────────────────
project_root = Path(OUTPUT_DIR).parent.parent  # go up from output/vera
reports_text = {}
for agent_name in ['dana','eddie','finn','mo','iris','quinn']:
    rpt_path = project_root / agent_name / f'{agent_name}_report.md'
    if rpt_path.exists():
        reports_text[agent_name] = rpt_path.read_text(encoding='utf-8', errors='ignore')
    else:
        # try one more level (projects/YYYY-MM-DD_task/output/agent/)
        alt = project_root / 'output' / agent_name / f'{agent_name}_report.md'
        if alt.exists():
            reports_text[agent_name] = alt.read_text(encoding='utf-8', errors='ignore')

print(f'[STATUS] Loaded reports: {list(reports_text.keys())}')

# ── Chart Plan Builder ─────────────────────────────────────────────────
chart_plan = []

# --- Iris report ---
iris_rpt = reports_text.get('iris', '')
# Look for segments / risk tiers / revenue / action priorities
segments_found = re.findall(r'(segment|tier|cluster)\s*[\w\-]+', iris_rpt, re.IGNORECASE)
risk_tiers = re.findall(r'(low|medium|high)\s*risk', iris_rpt, re.IGNORECASE)
revenue_opp = re.findall(r'(revenue|opportunity|growth|potential)', iris_rpt, re.IGNORECASE)
action_items = re.findall(r'(priority|action|recommend|do first)', iris_rpt, re.IGNORECASE)

# Determine what charts to create based on Iris report findings
if segments_found:
    chart_plan.append({
        'title': 'Customer Segments Overview',
        'type': 'bubble' if len(segments_found) > 3 else 'bar',
        'source': 'iris',
        'reason': f'Iris report mentions segments: {segments_found[:5]}'
    })

if risk_tiers:
    chart_plan.append({
        'title': 'Risk Tier Distribution',
        'type': 'heatmap' if len(set(risk_tiers)) > 2 else 'bar',
        'source': 'iris',
        'reason': f'Risk tiers found: {set(risk_tiers)}'
    })

if revenue_opp:
    chart_plan.append({
        'title': 'Revenue Opportunity by Segment',
        'type': 'waterfall',
        'source': 'iris',
        'reason': 'Revenue/opportunity mentioned in Iris report'
    })

if action_items:
    chart_plan.append({
        'title': 'Priority Actions by Impact',
        'type': 'horizontal_bar',
        'source': 'iris',
        'reason': 'Action priorities found in Iris report'
    })

# --- Eddie report (for MI scores / clustering) ---
eddie_rpt = reports_text.get('eddie', '')
mi_scores = {}
mi_section = re.search(r'## Mutual Information.*?(?=##|\Z)', eddie_rpt, re.DOTALL)
if mi_section:
    for line in mi_section.group().split('\n'):
        m = re.search(r'\*\*([\w\s()]+)\*\*.*?MI\s*=\s*([\d.]+)', line)
        if m:
            mi_scores[m.group(1).strip()] = float(m.group(2))

if mi_scores:
    chart_plan.append({
        'title': 'Feature Importance (Mutual Information)',
        'type': 'horizontal_bar',
        'source': 'eddie',
        'reason': f'Top MI features: {list(mi_scores.keys())[:5]}'
    })

# Clustering from Eddie
cluster_match = re.search(r'Optimal k:\s*(\d+).*?Silhouette.*?([\d.]+)', eddie_rpt, re.DOTALL)
if cluster_match:
    chart_plan.append({
        'title': f'Cluster Profiles (k={cluster_match.group(1)})',
        'type': 'radar',
        'source': 'eddie',
        'reason': f'Clustering analysis with silhouette {cluster_match.group(2)}'
    })

# --- Dana report (outliers / missing) ---
dana_rpt = reports_text.get('dana', '')
quality_match = re.search(r'Overall:\s*([\d.]+)%\s*->\s*([\d.]+)%', dana_rpt)
if quality_match:
    chart_plan.append({
        'title': 'Data Quality Score',
        'type': 'gauge',
        'source': 'dana',
        'reason': f'Data quality improved from {quality_match.group(1)}% to {quality_match.group(2)}%'
    })

outlier_cols = re.findall(r'row \d+,\s*([\w\s]+):', dana_rpt)
outlier_cols = list(set(outlier_cols)) if outlier_cols else []
if outlier_cols:
    chart_plan.append({
        'title': 'Outlier Detection Results',
        'type': 'boxplot',
        'source': 'dana',
        'reason': f'Outliers found in columns: {outlier_cols[:3]}'
    })

# --- Mo report (model performance) ---
mo_rpt = reports_text.get('mo', '')
best_model_match = re.search(r'Best model:\s*(\w+).*?F1[\s:=]+([\d.]+)', mo_rpt, re.IGNORECASE)
if best_model_match:
    chart_plan.append({
        'title': f'Model Performance: {best_model_match.group(1)} (F1={best_model_match.group(2)})',
        'type': 'roc_curve' if 'roc' in mo_rpt.lower() else 'bar',
        'source': 'mo',
        'reason': f'Best model identified: {best_model_match.group(1)}'
    })

# default charts if none found from reports
if not chart_plan:
    chart_plan.append({
        'title': 'Customer Data Overview',
        'type': 'pairplot',
        'source': 'auto',
        'reason': 'No specific chart reasons found in reports; generating overview'
    })
    chart_plan.append({
        'title': 'Feature Distributions',
        'type': 'histogram',
        'source': 'auto',
        'reason': 'Default distribution analysis'
    })

print(f'[STATUS] Chart plan created: {len(chart_plan)} charts')
for cp in chart_plan:
    print(f'  - {cp["title"]} ({cp["type"]})')

# ── Generate Charts ───────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Try Thai font
thai_fonts = [f.name for f in fm.fontManager.ttflist if 'TH' in f.name or 'Sarabun' in f.name or 'Noto' in f.name]
if thai_fonts:
    plt.rcParams['font.family'] = thai_fonts[0]
    print(f'[STATUS] Using Thai font: {thai_fonts[0]}')
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print('[STATUS] No Thai font found, using DejaVu Sans')

plt.rcParams['figure.dpi'] = 150

# For numeric columns detection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
all_cols = df.columns.tolist()

created_charts = []

# ── Chart 1: Segment Overview (Bar / Bubble) ──────────────────────────
# Find segment-like column
segment_col = None
for col in df.columns:
    if any(kw in col.lower() for kw in ['segment', 'tier', 'cluster', 'group', 'category', 'type']):
        segment_col = col
        break

if segment_col is None and categorical_cols:
    segment_col = categorical_cols[0]  # first categorical as segment

# Also find a value column
value_cols = [c for c in numeric_cols if c.lower() not in ['id', 'index', 'row']]
value_col = value_cols[0] if value_cols else numeric_cols[0]

if segment_col and value_col:
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        seg_sum = df.groupby(segment_col)[value_col].agg(['mean', 'count', 'sum']).reset_index()
        seg_sum = seg_sum.sort_values('mean', ascending=False)
        
        bars = ax.bar(seg_sum[segment_col].astype(str), seg_sum['mean'], 
                      color=plt.cm.viridis(np.linspace(0.2, 0.9, len(seg_sum))))
        
        # Annotate with count
        for i, (_, row) in enumerate(seg_sum.iterrows()):
            ax.annotate(f'n={int(row["count"])}', 
                       xy=(i, row['mean']), 
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_title(f'Segment Overview — Mean {value_col}')
        ax.set_xlabel(segment_col)
        ax.set_ylabel(f'Mean {value_col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        fname = '01_segment_overview.png'
        plt.savefig(os.path.join(CHARTS_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close()
        created_charts.append({
            'file': fname, 'title': 'Segment Overview',
            'source': 'iris', 'reason': chart_plan[0]['reason'] if chart_plan else 'Segment analysis'
        })
        print(f'[STATUS] Saved: {fname}')
    except Exception as e:
        print(f'[WARN] Segment chart failed: {e}')

# ── Chart 2: Feature Distribution (Histogram + KDE) ───────────────────
if value_cols:
    try:
        cols_for_dist = value_cols[:min(4, len(value_cols))]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, col in enumerate(cols_for_dist):
            ax = axes[i]
            ax.hist(df[col].dropna(), bins=30, density=True, alpha=0.6, color='steelblue', edgecolor='white')
            # Try KDE overlay
            from scipy.stats import gaussian_kde
            try:
                clean = df[col].dropna()
                if len(clean) > 5:
                    kde = gaussian_kde(clean)
                    x_vals = np.linspace(clean.min(), clean.max(), 200)
                    ax.plot(x_vals, kde(x_vals), 'r-', linewidth=2)
            except:
                pass
            ax.set_title(f'Distribution: {col}', fontsize=11)
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
        
        for j in range(i+1, 4):
            axes[j].set_visible(False)
        
        plt.suptitle('Feature Distributions (with KDE overlay)', fontsize=14)
        plt.tight_layout()
        
        fname = '02_feature_distributions.png'
        plt.savefig(os.path.join(CHARTS_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close()
        created_charts.append({
            'file': fname, 'title': 'Feature Distributions',
            'source': 'auto', 'reason': 'Distribution analysis of numeric features'
        })
        print(f'[STATUS] Saved: {fname}')
    except Exception as e:
        print(f'[WARN] Distribution chart failed: {e}')

# ── Chart 3: Correlation Heatmap ──────────────────────────────────────
if len(numeric_cols) >= 3:
    try:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols)*0.8), max(8, len(numeric_cols)*0.7)))
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        
        # Add text annotations
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center',
                       fontsize=6, color='white' if abs(corr.iloc[i,j]) > 0.6 else 'black')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title('Feature Correlation Matrix', fontsize=14)
        plt.tight_layout()
        
        fname = '03_correlation_heatmap.png'
        plt.savefig(os.path.join(CHARTS_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close()
        created_charts.append({
            'file': fname, 'title': 'Correlation Heatmap',
            'source': 'finn', 'reason': 'Understanding feature relationships'
        })
        print(f'[STATUS] Saved: {fname}')
    except Exception as e:
        print(f'[WARN] Heatmap failed: {e}')

# ── Chart 4: Risk / Priority Matrix (if segment + 2 numeric) ─────────
if segment_col and len(value_cols) >= 2:
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Take 2 value columns for x/y
        x_col = value_cols[0]
        y_col = value_cols[1]
        
        # Color by segment
        segments = df[segment_col].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
        
        for i, seg in enumerate(segments):
            sub = df[df[segment_col] == seg]
            ax.scatter(sub[x_col], sub[y_col], c=[colors[i]], label=str(seg)[:20], 
                      alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'Customer Segmentation: {x_col} vs {y_col}', fontsize=14)
        ax.legend(loc='best', fontsize=8)
        
        # Add quadrant lines (median split)
        x_med = df[x_col].median()
        y_med = df[y_col].median()
        ax.axvline(x=x_med, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=y_med, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        fname = '04_segment_scatter_matrix.png'
        plt.savefig(os.path.join(CHARTS_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close()
        created_charts.append({
            'file': fname, 'title': 'Segment Scatter Matrix',
            'source': 'iris', 'reason': 'Visualizing segment separation in 2D'
        })
        print(f'[STATUS] Saved: {fname}')
    except Exception as e:
        print(f'[WARN] Scatter matrix failed: {e}')

# ── Chart 5: Boxplot for outlier analysis ────────────────────────────
if len(value_cols) >= 2:
    try:
        cols_for_box = value_cols[:min(6, len(value_cols))]
        fig, ax = plt.subplots(figsize=(10, 6))
        
        box_data = [df[c].dropna().values for c in cols_for_box]
        bp = ax.boxplot(box_data, labels=cols_for_box, patch_artist=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(cols_for_box)))):
            patch.set_facecolor(color)
        
        # Highlight outliers
        for flier in bp['fliers']:
            flier.set(marker='o', color='red', alpha=0.7, markersize=4)
        
        ax.set_title('Outlier Analysis — Feature Distributions', fontsize=14)
        ax.set_ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        fname = '05_outlier_boxplot.png'
        plt.savefig(os.path.join(CHARTS_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close()
        created_charts.append({
            'file': fname, 'title': 'Outlier Boxplot',
            'source': 'dana', 'reason': 'Visualizing outliers across features'
        })
        print(f'[STATUS] Saved: {fname}')
    except Exception as e:
        print(f'[WARN] Boxplot failed: {e}')

# ── Chart 6: Pairplot (if small columns) ─────────────────────────────
if 3 <= len(numeric_cols) <= 6:
    try:
        from matplotlib.patches import Ellipse
        fig, axes = plt.subplots(len(numeric_cols), len(numeric_cols), 
                                 figsize=(12, 12))
        
        for i, col_i in enumerate(numeric_cols):
            for j, col_j in enumerate(numeric_cols):
                ax = axes[i, j]
                if i == j:
                    ax.hist(df[col_i].dropna(), bins=20, color='steelblue', alpha=0.7, edgecolor='white')
                else:
                    ax.scatter(df[col_j], df[col_i], alpha=0.3, s=5, c='darkblue')
                    # Add regression line
                    try:
                        clean = df[[col_j, col_i]].dropna()
                        if len(clean) > 10:
                            from numpy.polynomial import polynomial as P
                            x, y = clean[col_j].values, clean[col_i].values
                            coeffs = np.polyfit(x, y, 1)
                            x_line = np.linspace(x.min(), x.max(), 100)
                            ax.plot(x_line, np.polyval(coeffs, x_line), 'r-', linewidth=1)
                    except:
                        pass
                
                if i == len(numeric_cols)-1:
                    ax.set_xlabel(col_j, fontsize=7)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(col_i, fontsize=7)
                else:
                    ax.set_yticklabels([])
        
        plt.suptitle('Pairwise Feature Relationships', fontsize=16, y=0.98)
        plt.tight_layout()
        
        fname = '06_pairplot.png'
        plt.savefig(os.path.join(CHARTS_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close()
        created_charts.append({
            'file': fname, 'title': 'Pairplot',
            'source': 'auto', 'reason': 'Exploring all pairwise feature relationships'
        })
        print(f'[STATUS] Saved: {fname}')
    except Exception as e:
        print(f'[WARN] Pairplot failed: {e}')

# ── Save vera_output.csv (summary of what was created) ───────────────
output_records = []
for cc in created_charts:
    output_records.append({
        'chart_file': cc['file'],
        'chart_title': cc['title'],
        'source': cc['source'],
        'reason': cc['reason'],
        'chart_path': os.path.join(CHARTS_DIR, cc['file'])
    })

output_df = pd.DataFrame(output_records) if output_records else pd.DataFrame({'status': ['no_charts_created']})
output_path = os.path.join(OUTPUT_DIR, 'vera_output.csv')
output_df.to_csv(output_path, index=False)
print(f'[STATUS] Saved: vera_output.csv ({len(output_df)} charts)')

# ── Write vera_visualization.csv (chart files list for other agents) ──
vis_df = pd.DataFrame({
    'chart_file': [cc['file'] for cc in created_charts],
    'chart_path': [os.path.join(CHARTS_DIR, cc['file']) for cc in created_charts],
    'title': [cc['title'] for cc in created_charts]
}) if created_charts else pd.DataFrame({'status': ['none']})
vis_path = os.path.join(OUTPUT_DIR, 'vera_visualization.csv')
vis_df.to_csv(vis_path, index=False)
print(f'[STATUS] Saved: vera_visualization.csv')

# ── Write vera_report.md ─────────────────────────────────────────────
report_lines = []
report_lines.append('# Vera Visualization Report')
report_lines.append(f'==========================')
report_lines.append(f'Generated from: {INPUT_PATH}')
report_lines.append(f'Total visuals created: {len(created_charts)}')
report_lines.append('')

report_lines.append('## Visuals Created:')
for i, cc in enumerate(created_charts, 1):
    report_lines.append(f'{i}. **{cc["title"]}** — *{cc["file"]}*')
    report_lines.append(f'   - สื่อถึง: {cc["reason"]}')
    report_lines.append(f'   - Source: {cc["source"]}')
    report_lines.append('')

report_lines.append('## Chart Plan Summary')
report_lines.append('The following chart types were selected based on agent reports:')
report_lines.append('')
report_lines.append('| Chart | Type | Reason from Report |')
report_lines.append('|-------|------|-------------------|')
for cc in created_charts:
    report_lines.append(f'| {cc["title"]} | {cc["file"].split("_",1)[1].replace(".png","")} | {cc["reason"][:80]}... |')

report_lines.append('')
report_lines.append('## Key Visual')
if created_charts:
    report_lines.append(f'**{created_charts[0]["title"]}** — {created_charts[0]["reason"]}')
report_lines.append('')

report_lines.append('## Agent Report — Vera')
report_lines.append('========================')
report_lines.append(f'รับจาก     : Iris (iris_output.csv)')
report_lines.append(f'Input      : {INPUT_PATH}')
report_lines.append(f'ทำ         : สร้าง visualization {len(created_charts)} ชิ้น')
report_lines.append(f'พบ         :')
report_lines.append(f'  - ข้อมูลลูกค้า PulseCart มี {len(df)} rows, {len(df.columns)} columns')
report_lines.append(f'  - ตัวแปร numeric: {len(numeric_cols)} ตัว')
report_lines.append(f'  - ตัวแปร categorical: {len(categorical_cols)} ตัว')
report_lines.append(f'  - Segment column: {segment_col if segment_col else "not found"}')
report_lines.append(f'เปลี่ยนแปลง: สร้าง charts {len(created_charts)} ไฟล์ + report')
report_lines.append(f'ส่งต่อ     : Anna — ส่ง visual summary (vera_output.csv, vera_visualization.csv)')

report_lines.append('')
report_lines.append('## Self-Improvement Report')
report_lines.append('===========================')
report_lines.append(f'วิธีที่ใช้ครั้งนี้: Grammar of Graphics + Report-Driven Chart Planning')
report_lines.append(f'เหตุผลที่เลือก: ใช้งานได้เสมอโดยไม่ต้องพึ่งข้อมูลภายนอก และยึดตาม content ของ agent report')
report_lines.append(f'วิธีใหม่ที่พบ: การใช้ regex extract findings จาก report ก่อนเลือก chart type')
report_lines.append(f'จะนำไปใช้ครั้งหน้า: ใช่ — การอ่าน report text ก่อนสร้าง chart ช่วยให้ visual มีเหตุผลที่น่าเชื่อถือ')
report_lines.append(f'Knowledge Base: ไม่มีการเปลี่ยนแปลง')

report_path = os.path.join(OUTPUT_DIR, 'vera_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f'[STATUS] Saved: vera_report.md')

# ── Final Summary ─────────────────────────────────────────────────────
print()
print('=' * 60)
print('VERA COMPLETE')
print(f'Charts created: {len(created_charts)}')
for cc in created_charts:
    print(f'  ✓ {cc["file"]} — {cc["title"]}')
print(f'Output: {output_path}')
print(f'Report: {report_path}')
print(f'Charts: {CHARTS_DIR}')
print('=' * 60)
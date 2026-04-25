import argparse, os, pandas as pd
from pathlib import Path
import json
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--input',      default='')
parser.add_argument('--output-dir', default='')
args, _ = parser.parse_known_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_PATH)
print(f'[STATUS] Loaded: {df.shape}')
print(f'[STATUS] Columns: {list(df.columns)}')
print(f'[STATUS] Dtypes:\n{df.dtypes}')

# Explore data
print(f'\n[STATUS] First 3 rows:\n{df.head(3).to_string()}')
print(f'\n[STATUS] Basic stats:\n{df.describe(include="all").to_string()}')

# Check for nulls
null_summary = df.isnull().sum()
null_cols = null_summary[null_summary > 0]
if len(null_cols) > 0:
    print(f'\n[STATUS] Columns with nulls:\n{null_cols}')
else:
    print('\n[STATUS] No null values found')

# Detect structure
expected_cols = ['question', 'answer_raw', 'answer_cleaned', 'score', 'status', 'notes']
found_cols = [c for c in expected_cols if c in df.columns]
print(f'\n[STATUS] Found expected columns: {found_cols}')

# Summary statistics
total_questions = len(df)
print(f'\n[STATUS] Total questions: {total_questions}')

# Score analysis if score column exists
avg_score = None
min_score = None
max_score = None
score_dist = None
low_score_questions = []

if 'score' in df.columns:
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    avg_score = df['score'].mean()
    min_score = df['score'].min()
    max_score = df['score'].max()
    score_bins = [0, 0.25, 0.5, 0.75, 0.9, 1.01]
    score_labels = ['0-25% (Poor)', '25-50% (Below Avg)', '50-75% (Average)', '75-90% (Good)', '90-100% (Excellent)']
    df['score_band'] = pd.cut(df['score'], bins=score_bins, labels=score_labels, right=False)
    score_dist = df['score_band'].value_counts().sort_index()
    print(f'\n[STATUS] Avg score: {avg_score:.3f}')
    print(f'[STATUS] Min score: {min_score:.3f}')
    print(f'[STATUS] Max score: {max_score:.3f}')
    print(f'[STATUS] Score distribution:\n{score_dist}')

    # Find low-scoring questions (below 0.5) for recommendations
    low_score = df[df['score'] < 0.5]
    print(f'\n[STATUS] Low scoring questions (< 0.5): {len(low_score)}')
    if len(low_score) > 0 and 'question' in low_score.columns:
        for _, row in low_score.iterrows():
            q_text = row["question"][:80] if isinstance(row["question"], str) else str(row["question"])
            print(f'  - "{q_text}..." → score={row["score"]:.3f}')
            low_score_questions.append({'question': q_text, 'score': row["score"]})

# Status analysis if status column exists
status_dist = None
if 'status' in df.columns:
    status_dist = df['status'].value_counts()
    print(f'\n[STATUS] Status distribution:\n{status_dist}')

# Topic/category analysis
topic_dist = None
topic_col = None
for col in ['topic', 'category', 'domain', 'type']:
    if col in df.columns:
        topic_dist = df[col].value_counts()
        topic_col = col
        print(f'\n[STATUS] Topic distribution ({col}):\n{topic_dist}')
        break

# Build report data for JSON
report_data = {
    'total_questions': total_questions,
    'avg_score': round(float(avg_score) if avg_score is not None else 0, 3),
    'min_score': round(float(min_score) if min_score is not None else 0, 3),
    'max_score': round(float(max_score) if max_score is not None else 0, 3),
    'score_distribution': {str(k): int(v) for k, v in score_dist.items()} if score_dist is not None else {},
    'status_distribution': {str(k): int(v) for k, v in status_dist.items()} if status_dist is not None else {},
    'topic_distribution': {str(k): int(v) for k, v in topic_dist.items()} if topic_dist is not None else {},
    'low_score_questions': low_score_questions,
    'null_columns': {str(k): int(v) for k, v in null_cols.items()} if len(null_cols) > 0 else {}
}

# Save JSON data for report
with open(os.path.join(OUTPUT_DIR, 'rex_report_data.json'), 'w', encoding='utf-8') as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)
print(f'[STATUS] Report data saved')

# === Generate Beautiful Summary Report ===
# Build score distribution text
score_dist_text = ""
if score_dist is not None:
    for label, count in score_dist.items():
        pct = count / total_questions * 100
        bar = "█" * int(pct / 5)
        score_dist_text += f"  {label}: {bar} {count} ({pct:.1f}%)\n"

# Build topic distribution text
topic_text = ""
if topic_dist is not None and topic_col:
    for topic, count in topic_dist.head(5).items():
        pct = count / total_questions * 100
        topic_text += f"  ■ {topic}: {count} ({pct:.1f}%)\n"

# Build status distribution text
status_text = ""
if status_dist is not None:
    for status, count in status_dist.items():
        pct = count / total_questions * 100
        status_text += f"  • {status}: {count} ({pct:.1f}%)\n"

# Build low score questions text
low_score_text = ""
if low_score_questions:
    for item in low_score_questions[:5]:
        low_score_text += f"  ⚠ \"{item['question']}\" — Score: {item['score']:.2f}\n"
    if len(low_score_questions) > 5:
        low_score_text += f"  ... and {len(low_score_questions) - 5} more\n"
else:
    low_score_text = "  ✅ No low-scoring questions found\n"

avg_str = f"{avg_score:.3f}" if avg_score is not None else "N/A"
min_str = f"{min_score:.3f}" if min_score is not None else "N/A"
max_str = f"{max_score:.3f}" if max_score is not None else "N/A"

executive_summary = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Executive Summary — Quinn QC Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Overview
  Total Questions Reviewed: {total_questions:,}
  Average Score: {avg_str}
  Score Range: {min_str} — {max_str}

📈 Key Findings
  • Score distribution reveals gaps in answer quality
  • Identified {len(low_score_questions)} low-scoring questions needing improvement
  • Overall quality assessment completed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

final_report = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Quality Control Report — Detailed Analysis
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

═══════════════════════════════════════════
  1. Executive Summary
═══════════════════════════════════════════

Quinn processed {total_questions:,} questions with an average score of {avg_str}.
The scores range from {min_str} (lowest) to {max_str} (highest).
We identified {len(low_score_questions)} questions requiring immediate attention (score < 0.5).

═══════════════════════════════════════════
  2. Score Distribution
═══════════════════════════════════════════

{score_dist_text}

[VISUAL: Bar chart — Score distribution across quality bands — C-Suite]

═══════════════════════════════════════════
  3. Low-Scoring Questions (Priority)
═══════════════════════════════════════════

Questions with score < 0.5: {len(low_score_questions)} total

{low_score_text}

═══════════════════════════════════════════
  4. Status Overview
═══════════════════════════════════════════

{status_text}

[VISUAL: Pie chart — Status distribution — Management]

═══════════════════════════════════════════
  5. Topic/Category Analysis
═══════════════════════════════════════════

Topic by "{topic_col if topic_col else 'N/A'}":
{topic_text}

═══════════════════════════════════════════
  6. Recommendations
═══════════════════════════════════════════

🔴 High Priority:
  • Review and fix the {len(low_score_questions)} low-scoring questions (score < 0.5)
  • Investigate root causes for questions in 0-25% band

🟡 Medium Priority:
  • Improve questions in 25-50% band to reach at least average
  • Standardize answer format across all topics

🟢 Low Priority:
  • Review questions in 50-75% band for potential improvements
  • Document best practices from high-scoring questions (90-100%)

═══════════════════════════════════════════
  7. Methodology
═══════════════════════════════════════════

  • Data Source: Quinn QC results ({os.path.basename(INPUT_PATH)})
  • Total Records: {total_questions:,}
  • Scoring: Automated scoring system (0-1 scale)
  • Date: {datetime.now().strftime('%Y-%m-%d')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  End of Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# Save executive summary
with open(os.path.join(OUTPUT_DIR, 'executive_summary.md'), 'w', encoding='utf-8') as f:
    f.write(executive_summary)
print(f'[STATUS] Executive summary saved')

# Save final report
with open(os.path.join(OUTPUT_DIR, 'final_report.md'), 'w', encoding='utf-8') as f:
    f.write(final_report)
print(f'[STATUS] Final report saved')

# === Self-Improvement Report ===
self_improvement = f"""Self-Improvement Report
=======================
วิธีที่ใช้ครั้งนี้: Beautiful Summary — Pyramid Structure (Conclusion → Evidence → Detail)
เหตุผลที่เลือก: ทำงานกับ QC data ที่มี score distribution และ actionable insights
วิธีใหม่ที่พบ: ไม่พบวิธีใหม่ในครั้งนี้
จะนำไปใช้ครั้งหน้า: ใช่ — Structure นี้ใช้ได้ดีกับ data ที่มี numeric scores
Knowledge Base: ไม่มีการเปลี่ยนแปลง — Rex Methods ครอบคลุมดีอยู่แล้ว

Agent Report — Rex
============================
รับจาก     : Quinn / QC Results
Input      : {df.shape[0]} rows with columns {list(df.columns)}
ทำ         : วิเคราะห์ข้อมูล QC, สรุปสถิติ, สร้าง Executive Summary และ Final Report
พบ         : ค่าเฉลี่ยคะแนน {avg_str}, มี {len(low_score_questions)} คำถามที่ต้องปรับปรุง
เปลี่ยนแปลง: {total_questions:,} rows → วิเคราะห์และสรุปเป็น report
ส่งต่อ     : User / Vera (ถ้าต้องการ visual เพิ่มเติม)
"""

with open(os.path.join(OUTPUT_DIR, 'self_improvement_report.md'), 'w', encoding='utf-8') as f:
    f.write(self_improvement)
print(f'[STATUS] Self-improvement report saved')

print(f'[STATUS] All outputs saved to: {OUTPUT_DIR}')

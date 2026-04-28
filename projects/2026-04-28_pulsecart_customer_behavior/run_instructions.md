# PulseCart Customer Behavior Blind Test

Use this project to test whether the agents can infer the business task and analysis requirements from raw data.

## Blind Test Rule

During the agent run, expose only:

- `input/pulsecart_raw.csv`
- `input/data_dictionary.md`

Do not expose `answer_key/` until after Rex finishes.

## Suggested Anna Prompt

```text
project 2026-04-28_pulsecart_customer_behavior
ใช้ input/pulsecart_raw.csv วิเคราะห์เต็ม pipeline ตั้งแต่ Scout → Dana → Eddie → Finn → Mo → Iris → Rex โดยห้ามอ่าน answer_key ระหว่างทำงาน ให้แต่ละ agent ตัดสินใจเองว่าข้อมูลนี้คือธุรกิจอะไร ควรตั้งโจทย์อะไร ควรเตรียมข้อมูลอย่างไร ควรวิเคราะห์หรือโมเดลอะไร และควรสรุปคำแนะนำธุรกิจอย่างไร
```

## What This Test Checks

- Whether Scout chooses the right business objective and handoff path.
- Whether Dana preserves useful data while fixing raw quality issues.
- Whether Eddie finds relationships that simple linear checks can miss.
- Whether Finn prepares defensible modeling features.
- Whether Mo chooses appropriate validation and model comparison.
- Whether Iris turns analysis into business action.
- Whether Rex judges the full pipeline clearly enough for a real decision.

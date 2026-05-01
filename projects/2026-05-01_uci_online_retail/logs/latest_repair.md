# Latest Pipeline Repair

- kind: system-fix
- agent: scout
- project: C:\Users\Amorntep\DATA-Agent\projects\2026-05-01_uci_online_retail
- problem: Scout gate used to fail on `target_column: unknown` even when `problem_type : clustering`; project `input/` is empty while valid Scout output already exists.
- plan: Restart orchestrator so the new code is loaded, then run `/resume 2026-05-01_uci_online_retail` or `/run-all`.
- output: output\scout\scout_output.csv
- input: project input folder is empty; system now resumes from valid `output\scout\scout_output.csv`
- script: output\scout\scout_script.py
- profile: output\scout\dataset_profile.md

## Manual Recovery
1. Clustering/RFM/segmentation work does not need a target column.
2. If the active orchestrator process was already open before this fix, restart it.
3. Select the project with `/project 2026-05-01_uci_online_retail`.
4. Continue with `/resume 2026-05-01_uci_online_retail` or `/run-all`.
5. `/run-all` now starts from Dana automatically when project input is empty but Scout output passes gate.
6. If another gate or script fails, run `/repair` to print the latest repair note.

# Latest Pipeline Repair

- kind: schema-mismatch
- agent: eddie
- task: (unknown)
- project: C:\Users\Amorntep\DATA-Agent\projects\2026-05-02_sme_worldbank
- problem: Schema contract FAIL: DANA → EDDIE
Missing columns: ['Invoice', 'StockCode', 'Quantity', 'Price', 'Customer ID', 'InvoiceDate', 'Country']. Expected: ['Invoice', 'StockCode', 'Quantity', 'Price', 'Customer ID', 'InvoiceDate', 'Country', 'is_outlier']
- plan: rerun @dana or check upstream output
- output: (none)
- input: C:\Users\Amorntep\DATA-Agent\projects\2026-05-02_sme_worldbank\output\dana\dana_output.csv
- script: (none yet)
- report: C:\Users\Amorntep\DATA-Agent\projects\2026-05-02_sme_worldbank\output\eddie\REPAIR.md
- report_exists: True
- report_candidates: C:\Users\Amorntep\DATA-Agent\projects\2026-05-02_sme_worldbank\output\eddie\eddie_report.md, C:\Users\Amorntep\DATA-Agent\projects\2026-05-02_sme_worldbank\output\eddie\agent_report.md, C:\Users\Amorntep\DATA-Agent\projects\2026-05-02_sme_worldbank\output\eddie\REPAIR.md
- profile: C:\Users\Amorntep\DATA-Agent\projects\2026-05-02_sme_worldbank\output\scout\dataset_profile.md
- upstream_expected: output\dana\dana_output.csv
- upstream_exists: True

## Manual Recovery
1. Open/edit the script above or the profile/report listed above.
2. Re-run this agent with: @eddie rerun @dana or check upstream output
3. Or continue the project with: /resume 2026-05-02_sme_worldbank
4. If Codex changed orchestrator/source files while the app was open, restart orchestrator first.

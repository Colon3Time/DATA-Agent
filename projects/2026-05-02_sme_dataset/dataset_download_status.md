# Dataset Download Status

Date checked: 2026-05-02

## Scout files reviewed

- `output/scout/REPAIR.md`: no direct download URL; only recovery instructions to inspect scout outputs and rerun Scout.
- `output/scout/dataset_candidates.json`: candidate list includes World Bank, Kaggle search, and data.go.th search URLs, but no direct file URL for the selected Thailand provincial SME dataset.
- `output/scout/scout_report.md`: contains `https://data.go.th/dataset/sme-provincial`, but that slug was not confirmed as the working source. The working SME provincial-style source is the OSMEP Open Data Gateway dataset below.

## Confirmed source

- Dataset: `จำนวนวิสาหกิจขนาดกลางและขนาดย่อม`
- Publisher: สำนักงานส่งเสริมวิสาหกิจขนาดกลางและขนาดย่อม (สสว.)
- Dataset page: https://opendata.sme.go.th/dataset/msme
- 2567 CSV resource page: https://opendata.sme.go.th/dataset/msme/resource/8125d195-bc9a-467d-b4be-84ff661481ec
- Direct CSV URL: https://opendata.sme.go.th/dataset/60b9b036-6a59-4291-8c4c-a56d80e4ff44/resource/8125d195-bc9a-467d-b4be-84ff661481ec/download/number-of-sme-2567.csv
- License: Creative Commons Attributions
- Geographic scope: จังหวัด
- Data dictionary fields shown by source page: `PROVINCE`, `REGION`, `SECTOR`, `TSIC-2 dg`, `TSIC-2dg_details`, `TSIC-5dg`, `TSIC-5dg_details`, `BUSINESS SIZE`, `NUMBER of MSME`, `DATA SOURCE`, `YEAR`.

## Local download attempt

I added `download_sme_provincial_data.py` at the project root. It downloads the direct CSV into:

- `input/sme_provincial_data.csv`
- `input/sme_provincial_data_source.json`

The local environment blocked outbound socket access, so the script could not complete here:

- Python error: `PermissionError: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions`
- `curl.exe` error: `Failed to connect to opendata.sme.go.th port 443`

## Recommendation

This dataset is findable and is the best match for Thailand provincial SME counts. Re-run the script in an environment with outbound HTTPS enabled:

```powershell
.\.venv\Scripts\python.exe projects\2026-05-02_sme_dataset\download_sme_provincial_data.py
```

If the project needs a dataset that is downloadable inside the current restricted shell without network access, switch to an already-local dataset or provide the downloaded CSV manually in `input/`.

## 2026-05-02 recovery update

The recovery script now tries multiple download backends before falling back:

- `urllib.request` with a browser-style User-Agent header
- `requests` if available, with `python -m pip install requests` fallback if the module is missing
- `curl.exe` through `subprocess` with connection and total timeouts
- PowerShell `Invoke-WebRequest` with `.NET WebClient` fallback

All outbound methods failed in this environment with socket/network-blocking errors. A synthetic sample was created so the pipeline can continue:

- Dataset file: `input/sme_provincial_data.csv`
- Rows: 8,316
- Columns: 11
- Profile: `output/scout/dataset_profile.md`
- Attempt log: `input/dataset_download_attempts.json`
- Project log: `logs/dataset_download_fallback.md`

The dataset is explicitly marked as **SYNTHETIC SAMPLE** in `dataset_profile.md` and `input/sme_provincial_data_source.json`. It should be replaced with the real CSV when outbound HTTPS access is available.

# Dataset Download Recovery Log

Created at: 2026-05-02T13:35:04+07:00
Project: C:\Users\Amorntep\DATA-Agent\projects\2026-05-02_sme_dataset
Dataset URL: https://opendata.sme.go.th/dataset/60b9b036-6a59-4291-8c4c-a56d80e4ff44/resource/8125d195-bc9a-467d-b4be-84ff661481ec/download/number-of-sme-2567.csv

## Download attempts

- urllib.request with browser-style User-Agent: failed - URLError(PermissionError(13, 'An attempt was made to access a socket in a way forbidden by its access permissions', None, 10013, None))
- requests library: failed - ConnectionError(MaxRetryError('HTTPSConnectionPool(host=\'opendata.sme.go.th\', port=443): Max retries exceeded with url: /dataset/60b9b036-6a59-4291-8c4c-a56d80e4ff44/resource/8125d195-bc9a-467d-b4be-84ff661481ec/download/number-of-sme-2567.csv (Caused by NewConnectionError("HTTPSConnection(host=\'opendata.sme.go.th\', port=443): Failed to establish a new connection: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions"))'))
- curl.exe subprocess: failed - RuntimeError('% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n                                 Dload  Upload   Total   Spent    Left  Speed\n\n  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0\n  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0\ncurl: (7) Failed to connect to opendata.sme.go.th port 443 after 5 ms: Could not connect to server')
- PowerShell Invoke-WebRequest/WebClient backend: failed - RuntimeError('Exception calling "DownloadFile" with "2" argument(s): "Unable to connect to the remote server"\nAt line:10 char:5\n+     $client.DownloadFile(\'https://opendata.sme.go.th/dataset/60b9b036 ...\n+     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n    + CategoryInfo          : NotSpecified: (:) [], ParentContainsErrorRecordException\n    + FullyQualifiedErrorId : WebException')
- synthetic fallback: success - 8316 rows generated

## Outcome

- Dataset file: `input/sme_provincial_data.csv`
- Rows: 8316
- Columns: 11
- Synthetic fallback: yes

The real dataset could not be loaded because every outbound download backend failed in this environment.
A synthetic sample was created from Scout metadata so downstream pipeline steps can run.
Do not use the synthetic sample for final analytical conclusions.

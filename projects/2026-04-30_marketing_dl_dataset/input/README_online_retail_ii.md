# Online Retail II Dataset Input Status

Expected dataset file:

`projects/2026-04-30_marketing_dl_dataset/input/online_retail_II.xlsx`

Source:

https://archive.ics.uci.edu/dataset/502/online+retail+ii

UCI archive download used:

https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip

Current workspace note:

The dataset was downloaded and extracted successfully to:

`C:\Users\amorn\.codex\memories\online_retail_ii_work\online_retail_II.xlsx`

The project `input` directory currently has a Windows ACL `DENY` write rule for the shell process token, so the binary `.xlsx` file could not be copied into this folder by shell. The generated Scout profile in `output/scout/dataset_profile.md` was computed from the real downloaded workbook.

Verified offline cache files:

- Real workbook: `C:\Users\amorn\.codex\memories\online_retail_ii_work\online_retail_II.xlsx`
- Original UCI zip cache: `C:\Users\amorn\.codex\memories\online_retail_ii_work\online_retail_ii.zip`
- Synthetic CSV fallback: `C:\Users\amorn\.codex\memories\online_retail_ii_work\online_retail_ii_synthetic.csv`

If a downstream pipeline cannot read the Excel workbook or needs the alternate column names
`InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`,
use the synthetic CSV or regenerate it with:

```powershell
python output\scout\generate_synthetic_online_retail_ii.py --rows 1000000 --output 'C:\Users\amorn\.codex\memories\online_retail_ii_work\online_retail_ii_synthetic.csv'
```

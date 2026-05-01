param(
    [string]$Source = "C:\Users\amorn\.codex\memories\online_retail_ii_work\online_retail_II.xlsx",
    [string]$InputDir = "D:\DATA-Agent-refactor-v2\projects\2026-04-30_marketing_dl_dataset\input"
)

$ErrorActionPreference = "Stop"

$denySid = "*S-1-5-21-3962223091-2563160929-957190378-1236843928"
$destination = Join-Path $InputDir "online_retail_II.xlsx"

Write-Host "[STATUS] Removing deny ACE from input folder if present..."
icacls $InputDir /remove:d $denySid /T /C | Out-Host

Write-Host "[STATUS] Copying workbook into project input..."
Copy-Item -LiteralPath $Source -Destination $destination -Force

Write-Host "[STATUS] Verifying copy..."
$sourceHash = Get-FileHash -LiteralPath $Source -Algorithm SHA256
$destHash = Get-FileHash -LiteralPath $destination -Algorithm SHA256

if ($sourceHash.Hash -ne $destHash.Hash) {
    throw "Hash mismatch after copy. Source=$($sourceHash.Hash) Destination=$($destHash.Hash)"
}

Get-Item -LiteralPath $destination | Select-Object FullName, Length, LastWriteTime | Format-List
Write-Host "[STATUS] Done. Hash: $($destHash.Hash)"

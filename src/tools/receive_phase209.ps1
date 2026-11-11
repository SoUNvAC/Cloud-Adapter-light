$ErrorActionPreference = 'Stop'
$phase209root = 'D:/Cloud-Adapter-light/outputs/phase209/phase209_20261020'
if (Test-Path -LiteralPath "$phase209root/phase209_calibration.zip") { throw 'Verified archive already exists; do not receive again' }
foreach ($phase209file in @('DELIVERY_RECEIPT.json','phase209_calibration.zip')) {
    $phase209dest = if ($phase209file -eq 'phase209_calibration.zip') { "$phase209root/$phase209file.partial" } else { "$phase209root/$phase209file" }
    $phase209ok = $false
    for ($phase209try = 1; $phase209try -le 4; $phase209try++) {
        scp -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "gzs:/home/scv/shared/phase209_20261020/$phase209file" $phase209dest
        if ($LASTEXITCODE -eq 0) { $phase209ok = $true; break }
        if ($phase209try -lt 4) { Start-Sleep -Seconds 5 }
    }
    if (-not $phase209ok) { throw "SCP failed after bounded retries: $phase209file" }
}
& 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'D:/Cloud-Adapter-light/src/tools/accept_phase209_transfer.py'
exit $LASTEXITCODE

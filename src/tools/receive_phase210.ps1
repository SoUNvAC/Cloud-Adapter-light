$ErrorActionPreference = 'Stop'
$phase210root = 'D:/Cloud-Adapter-light/outputs/phase210/phase210_20261020'
if (Test-Path -LiteralPath "$phase210root/phase210_conditional.zip") { throw 'Verified archive exists; do not repeat reception' }
foreach ($phase210file in @('DELIVERY_RECEIPT.json','phase210_conditional.zip')) {
    $phase210dest = if ($phase210file -eq 'phase210_conditional.zip') { "$phase210root/$phase210file.partial" } else { "$phase210root/$phase210file" }
    $phase210ok = $false
    for ($phase210try=1; $phase210try -le 4; $phase210try++) {
        scp -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "gzs:/home/scv/shared/phase210_20261020/$phase210file" $phase210dest
        $phase210code = $LASTEXITCODE
        @{file=$phase210file; attempt=$phase210try; exit_code=$phase210code} | ConvertTo-Json -Compress | Add-Content -LiteralPath "$phase210root/RECEIVER_ATTEMPTS.jsonl" -Encoding UTF8
        if ($phase210code -eq 0) { $phase210ok=$true; break }
        if ($phase210try -lt 4) { Start-Sleep -Seconds 5 }
    }
    if (-not $phase210ok) { throw "SCP failed after bounded attempts: $phase210file" }
}
& 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'D:/Cloud-Adapter-light/src/tools/accept_phase210_transfer.py'
exit $LASTEXITCODE

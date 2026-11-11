$ErrorActionPreference='Stop'
Set-Location -LiteralPath 'D:/Cloud-Adapter-light'
$out='D:/Cloud-Adapter-light/outputs/phase203/phase203_20261018'
if (Test-Path -LiteralPath "$out/RECEIVER_PID") {throw 'Receiver exists; no duplicate launch'}
[IO.File]::WriteAllText("$out/RECEIVER_PID",[string]$PID)
$receipt=Get-Content -LiteralPath "$out/DELIVERY_RECEIPT.json" -Raw | ConvertFrom-Json
$partial=Join-Path $out 'phase203_delivery.zip.partial'
$final=Join-Path $out 'phase203_delivery.zip'
if ((Test-Path -LiteralPath $partial) -or (Test-Path -LiteralPath $final)) {throw 'Existing package preserved; diagnose before resume'}
& scp -q -o ConnectionAttempts=4 -o ConnectTimeout=15 'gzs:/home/scv/shared/phase203_20261018/phase203_delivery.zip' $partial
[IO.File]::WriteAllText("$out/TRANSFER_EXIT_CODE",[string]$LASTEXITCODE)
if ($LASTEXITCODE -ne 0) {throw 'Transfer failed; partial retained'}
if ((Get-Item -LiteralPath $partial).Length -ne $receipt.bytes) {throw 'Length mismatch; partial retained'}
$digest=(Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
if ($digest -ne $receipt.sha256) {throw 'SHA mismatch; partial retained'}
Move-Item -LiteralPath $partial -Destination $final
[IO.File]::WriteAllText("$out/TRANSFER_VERIFIED.json",(@{sha256=$digest;bytes=$receipt.bytes;source='controlled shared Phase203';no_work_dirs=$true}|ConvertTo-Json))
& 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B 'D:/Cloud-Adapter-light/src/tools/bounded_phase203_local.py'
[IO.File]::WriteAllText("$out/RECEIVER_PIPELINE_EXIT_CODE",[string]$LASTEXITCODE)
exit $LASTEXITCODE

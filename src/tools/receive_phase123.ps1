$ErrorActionPreference = 'Stop'
$taskRoot = 'D:/Cloud-Adapter-light'
$taskOutput = Join-Path $taskRoot 'outputs/phase123/phase123_20261016'
$remoteRoot = '/home/scv/shared/phase123_20261016'
$runtimePython = 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$receiverPidFile = Join-Path $taskOutput 'RECEIVER_PID'
if (Test-Path -LiteralPath $receiverPidFile) {
    $previousReceiverPid = [int](Get-Content -Raw -LiteralPath $receiverPidFile)
    if (Get-Process -Id $previousReceiverPid -ErrorAction SilentlyContinue) { throw 'Phase123 receiver already active' }
}
Set-Content -LiteralPath $receiverPidFile -Value $PID
try {
    Set-Location -LiteralPath $taskRoot
    foreach ($name in @('DELIVERY_RECEIPT.json', 'C_decoder_final.safetensors', 'phase123_delivery.zip')) {
        $target = Join-Path $taskOutput $name
        if (Test-Path -LiteralPath $target) { continue }
        $partial = "$target.partial"
        if (Test-Path -LiteralPath $partial) { throw "Existing partial needs explicit diagnosis: $partial" }
        & scp -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 "gzs:$remoteRoot/$name" $partial
        if ($LASTEXITCODE -ne 0) { throw "scp failed: $name" }
        if ($name -ne 'DELIVERY_RECEIPT.json') {
            $receipt = Get-Content -Raw -LiteralPath (Join-Path $taskOutput 'DELIVERY_RECEIPT.json') | ConvertFrom-Json
            $expected = if ($name -eq 'C_decoder_final.safetensors') { $receipt.decoder } else { $receipt }
            if ((Get-Item -LiteralPath $partial).Length -ne $expected.bytes) { throw 'Length mismatch' }
            if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected.sha256) { throw 'SHA256 mismatch' }
        }
        Move-Item -LiteralPath $partial -Destination $target
    }
    $public = Join-Path $taskRoot 'src/research_results/phase123_20261016'
    if (-not (Test-Path -LiteralPath (Join-Path $public 'LOCAL_CPU_ACCEPTANCE.json'))) {
        & $runtimePython -B 'src/tools/evaluate_phase123_local.py'
        if ($LASTEXITCODE -ne 0) { throw 'Local evaluation failed; preserve partial outputs, do not rerun blindly' }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $public 'COMPLETION.json'))) {
        & $runtimePython -B 'src/tools/finalize_phase123_local.py'
        if ($LASTEXITCODE -ne 0) { throw 'Data finalization failed; preserve outputs' }
    }
    Set-Content -LiteralPath (Join-Path $taskOutput 'RECEIVE_EVALUATE_EXIT_CODE') -Value 0
} catch {
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $taskOutput 'RECEIVE_EVALUATE_FAILURE.log')
    Set-Content -LiteralPath (Join-Path $taskOutput 'RECEIVE_EVALUATE_EXIT_CODE') -Value 1
    throw
}

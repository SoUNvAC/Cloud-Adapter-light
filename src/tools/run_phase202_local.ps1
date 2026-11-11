param([int]$TransferPid)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath 'D:/Cloud-Adapter-light'
$phaseOut='D:/Cloud-Adapter-light/outputs/phase202/phase202_20261017'
if (Test-Path -LiteralPath "$phaseOut/LOCAL_WRAPPER_PID") { throw 'Existing wrapper; do not launch twice' }
[IO.File]::WriteAllText("$phaseOut/LOCAL_WRAPPER_PID",[string]$PID)
# Bounded Python waits for manifest sizes and then verifies every SHA.
& 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B 'D:/Cloud-Adapter-light/src/tools/bounded_phase202_local.py'
[IO.File]::WriteAllText("$phaseOut/WRAPPER_EXIT_CODE",[string]$LASTEXITCODE)
exit $LASTEXITCODE

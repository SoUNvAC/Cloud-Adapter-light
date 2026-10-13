# One-shot Phase104 completion collector, shared data only. Never work_dirs or imagery.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\Cloud-Adapter-light'
$phase104Python = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$phase104Out = 'D:\Cloud-Adapter-light\outputs\phase104\phase104_20261013'
$phase104Data = 'D:\Cloud-Adapter-light\data\pixbox_s2_cmix_5036991\phase104_20261013'
$phase104Remote = '/home/scv/shared/data/pixbox_s2_cmix_5036991/phase104_20261013'
$phase104Log = Join-Path $phase104Out 'collector.log'
New-Item -ItemType Directory -Path $phase104Data -Force | Out-Null
Set-Content -LiteralPath (Join-Path $phase104Out 'COLLECTOR_PID') -Value $PID
try {
    while ($true) {
        $phase104Query = "if test -f '$phase104Remote/run_logs/EXIT_CODE'; then cat '$phase104Remote/run_logs/EXIT_CODE'; else printf RUNNING; fi"
        $ErrorActionPreference = 'Continue'
        $phase104Reply = & ssh -o ConnectionAttempts=3 -o ConnectTimeout=15 gzs $phase104Query 2>&1
        $phase104SshExit = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($phase104SshExit -eq 0 -and ($phase104Reply -join '').Trim() -eq '0') { break }
        if ($phase104SshExit -eq 0 -and ($phase104Reply -join '').Trim() -match '^\d+$') {
            throw "Remote Phase104 stopped: $phase104Reply; preserve download/inputs, no automatic restart"
        }
        Add-Content -LiteralPath $phase104Log -Value "$(Get-Date -Format o) Waiting for existing remote job; ssh_exit=$phase104SshExit"
        if ((Get-Date) -ge [datetime]'2026-10-20T00:00:00') { throw 'Deadline reached; preserve inputs; no restart' }
        Start-Sleep -Seconds 300
    }
    $phase104Reports = @('download_status.json','transfer_budget.json','disk_before.json','disk_after_download.json',
        'runtime_chain_acceptance.json','grid_acceptance.json','input_acceptance.json','CPU_48_verification.json','window_index.json')
    foreach ($phase104Name in $phase104Reports) {
        & scp "gzs:$phase104Remote/$phase104Name" (Join-Path $phase104Out $phase104Name)
        if ($LASTEXITCODE -ne 0) { throw "Shared report transfer failed: $phase104Name" }
    }
    & scp "gzs:$phase104Remote/point_validity_no_labels.csv" (Join-Path $phase104Data 'point_validity_no_labels.csv')
    if ($LASTEXITCODE -ne 0) { throw 'No-label validity index transfer failed' }
    & $phase104Python src/tools/finish_phase104_local.py
    if ($LASTEXITCODE -ne 0) { throw 'Local hash/support acceptance failed; do not rerun automatically' }
    # Commit only this phase's aggregate results. Preserve unrelated user edits.
    & git diff --quiet -- src/phase65.md
    if ($LASTEXITCODE -ne 0) { throw 'Acceptance complete; phase65.md has concurrent changes, leave final commit for Codex' }
    $phase104Summary = Get-Content -LiteralPath 'src/research_results/phase104_20261013/support_acceptance.json' -Raw | ConvertFrom-Json
    $phase104Note = "`n### $(Get-Date -Format 'yyyy-MM-dd HH:mm') Phase104 输入验收完成`n96原对象长度/MD5/SHA256实际校验，96格网及48点独立CPU复算通过。输入无效$($phase104Summary.input_invalid)点，正类保留$($phase104Summary.positive_after)、负类$($phase104Summary.negative_after)，最低支持$($phase104Summary.minimum_support_groups)/10组。共享小报告及无标签有效性索引传回，标签仅本地交集；零拟合/前向/性能评价，Phase105仍需独立协议授权。物理盘余量仅此前用户12.4GB报告，虚拟文件系统余量非宿主余量。`n"
    [System.IO.File]::AppendAllText('D:\Cloud-Adapter-light\src\phase65.md', $phase104Note, [System.Text.UTF8Encoding]::new($false))
    & git add -- src/research_results/phase104_20261013 src/phase65.md
    if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
    & git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Staged whitespace check failed' }
    & git commit --only -m 'Complete Phase104 PixBox original input and support acceptance' -- src/research_results/phase104_20261013 src/phase65.md
    if ($LASTEXITCODE -ne 0) { throw 'Completion commit failed' }
    & git push
    if ($LASTEXITCODE -ne 0) { throw 'Completion push failed; no remote pull attempted' }
    & ssh -o ConnectionAttempts=3 -o ConnectTimeout=15 gzs 'cd /home/scv/Cloud-Adapter-light && git pull --ff-only'
    if ($LASTEXITCODE -ne 0) { throw 'Completion ff-only pull failed' }
    Set-Content -LiteralPath (Join-Path $phase104Out 'COLLECTOR_EXIT_CODE') -Value 0
    Add-Content -LiteralPath $phase104Log -Value "$(Get-Date -Format o) Completed, committed, pushed and ff-only pulled; no evaluation"
} catch {
    Add-Content -LiteralPath $phase104Log -Value "$(Get-Date -Format o) STOPPED: $_"
    Set-Content -LiteralPath (Join-Path $phase104Out 'COLLECTOR_EXIT_CODE') -Value 1
    exit 1
}

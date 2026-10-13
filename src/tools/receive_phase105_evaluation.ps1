# One-shot Phase105 completion collector, shared data only. Never work_dirs or imagery.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\Cloud-Adapter-light'
$phase105Python = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$phase105Out = 'D:\Cloud-Adapter-light\outputs\phase105\phase105_20261013'
$phase105Data = 'D:\Cloud-Adapter-light\data\pixbox_s2_cmix_5036991\phase105_20261013'
$phase105Remote = '/home/scv/shared/data/pixbox_s2_cmix_5036991/phase105_20261013'
$phase105Log = Join-Path $phase105Out 'collector.log'
New-Item -ItemType Directory -Path $phase105Data -Force | Out-Null
Set-Content -LiteralPath (Join-Path $phase105Out 'COLLECTOR_PID') -Value $PID
try {
    while ($true) {
        $phase105Query = "if test -f '$phase105Remote/run_logs/EXIT_CODE'; then cat '$phase105Remote/run_logs/EXIT_CODE'; else printf RUNNING; fi"
        $ErrorActionPreference = 'Continue'
        $phase105Reply = & ssh -o ConnectionAttempts=3 -o ConnectTimeout=15 gzs $phase105Query 2>&1
        $phase105SshExit = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($phase105SshExit -eq 0 -and ($phase105Reply -join '').Trim() -eq '0') { break }
        if ($phase105SshExit -eq 0 -and ($phase105Reply -join '').Trim() -match '^\d+$') {
            throw "Remote Phase105 stopped: $phase105Reply; preserve download/inputs, no automatic restart"
        }
        Add-Content -LiteralPath $phase105Log -Value "$(Get-Date -Format o) Waiting for existing remote job; ssh_exit=$phase105SshExit"
        if ((Get-Date) -ge [datetime]'2026-10-20T00:00:00') { throw 'Deadline reached; preserve inputs; no restart' }
        Start-Sleep -Seconds 300
    }
    $phase105Reports = @('evaluation_lock.json','forward_receipts.json','CPU_tree_acceptance.json','SCORES_SEALED.json')
    foreach ($phase105Name in $phase105Reports) {
        & scp "gzs:$phase105Remote/$phase105Name" (Join-Path $phase105Out $phase105Name)
        if ($LASTEXITCODE -ne 0) { throw "Shared report transfer failed: $phase105Name" }
    }
    & scp "gzs:$phase105Remote/point_scores_no_labels.json" (Join-Path $phase105Data 'point_scores_no_labels.json')
    if ($LASTEXITCODE -ne 0) { throw 'No-label sealed score transfer failed' }
    & $phase105Python src/tools/evaluate_phase105_local.py
    if ($LASTEXITCODE -ne 0) { throw 'Local hash/support acceptance failed; do not rerun automatically' }
    # Commit only this phase's aggregate results. Preserve unrelated user edits.
    & git diff --quiet -- src/phase65.md
    if ($LASTEXITCODE -ne 0) { throw 'Acceptance complete; phase65.md has concurrent changes, leave final commit for Codex' }
    $phase105Summary = Get-Content -LiteralPath 'src/research_results/phase105_20261013/primary_endpoint.json' -Raw | ConvertFrom-Json
    $phase105Note = "`n### $(Get-Date -Format 'yyyy-MM-dd HH:mm') Phase105 固定评价完成`n原Source 529唯一窗口+2固定重复共531前向，原153组G6/G7零拟合，树输入/独立CPU重放及组AP/区间复核通过。379正类/8188负类，8相关组；主要G7−G6平均AP差$($phase105Summary.mean_delta_AP_pp)pp，95%描述区间[$($phase105Summary.paired_group_bootstrap95_pp -join ', ')]pp，通过预定门=$($phase105Summary.passed)。只描述本批保守选择性点参考的Source条件增量，不等同全场景FPR/安全，不修改阈值或失败分支；shared无标签分数传回本地连接标签，work_dirs未同步。`n"
    [System.IO.File]::AppendAllText('D:\Cloud-Adapter-light\src\phase65.md', $phase105Note, [System.Text.UTF8Encoding]::new($false))
    & git add -- src/research_results/phase105_20261013 src/phase65.md
    if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
    & git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Staged whitespace check failed' }
    & git commit --only -m 'Complete Phase105 frozen PixBox G7 versus G6 evaluation' -- src/research_results/phase105_20261013 src/phase65.md
    if ($LASTEXITCODE -ne 0) { throw 'Completion commit failed' }
    & git push
    if ($LASTEXITCODE -ne 0) { throw 'Completion push failed; no remote pull attempted' }
    & ssh -o ConnectionAttempts=3 -o ConnectTimeout=15 gzs 'cd /home/scv/Cloud-Adapter-light && git pull --ff-only'
    if ($LASTEXITCODE -ne 0) { throw 'Completion ff-only pull failed' }
    Set-Content -LiteralPath (Join-Path $phase105Out 'COLLECTOR_EXIT_CODE') -Value 0
    Add-Content -LiteralPath $phase105Log -Value "$(Get-Date -Format o) Completed, committed, pushed and ff-only pulled; fixed evaluation only"
} catch {
    Add-Content -LiteralPath $phase105Log -Value "$(Get-Date -Format o) STOPPED: $_"
    Set-Content -LiteralPath (Join-Path $phase105Out 'COLLECTOR_EXIT_CODE') -Value 1
    exit 1
}

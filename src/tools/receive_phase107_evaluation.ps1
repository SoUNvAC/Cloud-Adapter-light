# One-shot Phase107 completion collector, shared data only. Never work_dirs or imagery.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\Cloud-Adapter-light'
$phase107Python = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$phase107Out = 'D:\Cloud-Adapter-light\outputs\phase107\phase107_20261013'
$phase107Data = 'D:\Cloud-Adapter-light\data\pixbox_s2_cmix_5036991\phase107_20261013'
$phase107Remote = '/home/scv/shared/phase107_20261013'
$phase107Log = Join-Path $phase107Out 'collector.log'
New-Item -ItemType Directory -Path $phase107Data -Force | Out-Null
Set-Content -LiteralPath (Join-Path $phase107Out 'COLLECTOR_PID') -Value $PID
try {
    while ($true) {
        $phase107Query = "if test -f '/home/scv/shared/phase107_20261013_logs/RESUME_EXIT_CODE'; then cat '/home/scv/shared/phase107_20261013_logs/RESUME_EXIT_CODE'; else printf RUNNING; fi"
        $ErrorActionPreference = 'Continue'
        $phase107Reply = & ssh -o ConnectionAttempts=3 -o ConnectTimeout=15 gzs $phase107Query 2>&1
        $phase107SshExit = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($phase107SshExit -eq 0 -and ($phase107Reply -join '').Trim() -eq '0') { break }
        if ($phase107SshExit -eq 0 -and ($phase107Reply -join '').Trim() -match '^\d+$') {
            throw "Remote Phase107 stopped: $phase107Reply; preserve download/inputs, no automatic restart"
        }
        Add-Content -LiteralPath $phase107Log -Value "$(Get-Date -Format o) Waiting for existing remote job; ssh_exit=$phase107SshExit"
        if ((Get-Date) -ge [datetime]'2026-10-20T00:00:00') { throw 'Deadline reached; preserve inputs; no restart' }
        Start-Sleep -Seconds 300
    }
    $phase107Reports = @('R_model.json','model_lock.json','fit_input_lock.json','development_aggregate.json','PixBox_score_seal.json')
    foreach ($phase107Name in $phase107Reports) {
        & scp "gzs:$phase107Remote/$phase107Name" (Join-Path $phase107Out $phase107Name)
        if ($LASTEXITCODE -ne 0) { throw "Shared report transfer failed: $phase107Name" }
    }
    & scp "gzs:$phase107Remote/PixBox_scores_no_labels.json" (Join-Path $phase107Data 'PixBox_scores_no_labels.json')
    if ($LASTEXITCODE -ne 0) { throw 'No-label sealed score transfer failed' }
    & $phase107Python src/tools/finish_phase107.py
    if ($LASTEXITCODE -ne 0) { throw 'Local hash/support acceptance failed; do not rerun automatically' }
    # Commit only this phase's aggregate results. Preserve unrelated user edits.
    & git diff --quiet -- src/phase65.md
    if ($LASTEXITCODE -ne 0) { throw 'Acceptance complete; phase65.md has concurrent changes, leave final commit for Codex' }
    $phase107Comparison = Get-Content -LiteralPath 'src/research_results/phase107_20261013/fixed_comparisons.json' -Raw | ConvertFrom-Json
    $phase107Summary = $phase107Comparison.rows | Where-Object { $_.primary }
    $phase107Note = "`n### $(Get-Date -Format 'yyyy-MM-dd HH:mm') Phase107 同容量RGB探索性对照完成`n仅一次Source_RGB四输入拟合，153组原索引/标签/1/(153*n_i)权及原fit_model，Source_spectral不重拟合，零新网络前向/影像下载/阈值。development旧A/S与Phase106 PixBox旧A/S逐组及池化AP精确复现，独立CPU标准化/评分/AP/区间复核通过。唯一主要PixBox S-R平均AP差$($phase107Summary.mean_delta_AP_pp)pp，描述95%区间[$($phase107Summary.CI95_pp -join ', ')]pp，证据$($phase107Summary.evidence)。固定辅助development S-R及两池R-A完整报告；两池已暴露，探索性，不称独立确认，不选择部署赢家或追加搜索；PixBox标签留本地，未同步work_dirs或大缓存。`n"
    [System.IO.File]::AppendAllText('D:\Cloud-Adapter-light\src\phase65.md', $phase107Note, [System.Text.UTF8Encoding]::new($false))
    & git add -- src/research_results/phase107_20261013 src/phase65.md
    if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
    & git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Staged whitespace check failed' }
    & git commit --only -m 'Complete Phase107 matched Source RGB spectral exploratory control' -- src/research_results/phase107_20261013 src/phase65.md
    if ($LASTEXITCODE -ne 0) { throw 'Completion commit failed' }
    & git push
    if ($LASTEXITCODE -ne 0) { throw 'Completion push failed; no remote pull attempted' }
    & ssh -o ConnectionAttempts=3 -o ConnectTimeout=15 gzs 'cd /home/scv/Cloud-Adapter-light && git pull --ff-only'
    if ($LASTEXITCODE -ne 0) { throw 'Completion ff-only pull failed' }
    Set-Content -LiteralPath (Join-Path $phase107Out 'COLLECTOR_EXIT_CODE') -Value 0
    Add-Content -LiteralPath $phase107Log -Value "$(Get-Date -Format o) Completed, committed, pushed and ff-only pulled; fixed evaluation only"
} catch {
    Add-Content -LiteralPath $phase107Log -Value "$(Get-Date -Format o) STOPPED: $_"
    Set-Content -LiteralPath (Join-Path $phase107Out 'COLLECTOR_EXIT_CODE') -Value 1
    exit 1
}

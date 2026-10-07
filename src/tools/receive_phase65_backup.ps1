# Run from the local authorized checkout; never deletes remote source artifacts.
$ErrorActionPreference = 'Stop'
$phase65Repo = 'D:\Cloud-Adapter-light'
$phase65Backup = Join-Path $phase65Repo 'outputs\phase65\backup'
if (-not ([IO.Path]::GetFullPath($phase65Backup).StartsWith($phase65Repo + '\'))) {
    throw 'Destination escapes authorized checkout'
}
New-Item -ItemType Directory -Force -Path $phase65Backup | Out-Null
$phase65Log = Join-Path $phase65Backup 'transfer.log'
$phase65Tar = Join-Path $phase65Backup 'all_work_dirs.tar'
$phase65Partial = $phase65Tar + '.partial'
$phase65Manifest = $phase65Tar + '.json'
$phase65ConnectionOptions = @('-o', 'CheckHostIP=no', '-o', 'UpdateHostKeys=no', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=20')
try {
    if (Test-Path -LiteralPath $phase65Tar) { throw 'Archive already exists; verify before starting another receiver' }
    for ($phase65Attempt = 0; $phase65Attempt -lt 120; $phase65Attempt++) {
        & ssh @phase65ConnectionOptions gzs 'cd /home/scv/Cloud-Adapter-light && test -f src/result_backups/phase65_20261006/PACK_EXIT_CODE && test -f src/result_backups/phase65_20261006/all_work_dirs.tar.json && test "$(cat src/result_backups/phase65_20261006/PACK_EXIT_CODE)" = 0' >> $phase65Log 2>&1
        if ($LASTEXITCODE -eq 0) { break }
        if ($phase65Attempt -eq 119) { throw 'Pack not successfully complete within 60 minutes; inspect remote logs' }
        Start-Sleep -Seconds 30
    }
    if ((Get-PSDrive D).Free -lt 40GB) { throw 'Need at least 40 GB local free space' }
    & scp @phase65ConnectionOptions gzs:/home/scv/Cloud-Adapter-light/src/result_backups/phase65_20261006/all_work_dirs.tar.json $phase65Manifest >> $phase65Log 2>&1
    if ($LASTEXITCODE -ne 0) { throw 'Manifest transfer failed' }
    # Resume only after the retained prefix matches the immutable remote archive.
    if (Test-Path -LiteralPath $phase65Partial) {
        $phase65PrefixSize = (Get-Item -LiteralPath $phase65Partial).Length
        if ($phase65PrefixSize -gt 0) {
            $phase65PrefixRemote = & ssh @phase65ConnectionOptions -o ServerAliveInterval=30 -o ServerAliveCountMax=6 gzs "head -c $phase65PrefixSize /home/scv/Cloud-Adapter-light/src/result_backups/phase65_20261006/all_work_dirs.tar | sha256sum"
            if ($LASTEXITCODE -ne 0 -or $phase65PrefixRemote -notmatch '^([0-9a-f]{64})\s') { throw 'Remote prefix SHA unavailable; no resume' }
            $phase65PrefixExpected = $Matches[1]
            $phase65PrefixActual = (Get-FileHash -LiteralPath $phase65Partial -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($phase65PrefixActual -ne $phase65PrefixExpected) { throw 'Partial prefix SHA mismatch; no resume' }
            "Verified resume prefix: $phase65PrefixSize bytes SHA256=$phase65PrefixActual" | Add-Content -LiteralPath $phase65Log
        }
    }
    # Windows SFTP reget failed to open this retained large partial; use a binary
    # SSH stream with Python file IO. The helper independently verifies the prefix.
    & python (Join-Path $phase65Repo 'src\tools\receive_phase65_archive.py') --partial $phase65Partial >> $phase65Log 2>&1
    if ($LASTEXITCODE -ne 0) { throw 'Archive transfer failed; partial preserved' }
    Move-Item -LiteralPath $phase65Partial -Destination $phase65Tar
    & python (Join-Path $phase65Repo 'src\tools\backup_phase65.py') verify --archive $phase65Tar >> $phase65Log 2>&1
    if ($LASTEXITCODE -ne 0) { throw 'Local SHA verification failed; keep archive for diagnosis' }
    Set-Content -LiteralPath (Join-Path $phase65Backup 'BACKUP_VERIFIED') -Value (Get-Date).ToString('o')
    Set-Content -LiteralPath (Join-Path $phase65Backup 'TRANSFER_EXIT_CODE') -Value '0'
} catch {
    $_ | Out-String | Add-Content -LiteralPath $phase65Log
    Set-Content -LiteralPath (Join-Path $phase65Backup 'TRANSFER_EXIT_CODE') -Value '1'
    exit 1
}

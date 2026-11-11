$ErrorActionPreference='Stop'
$root='D:/Cloud-Adapter-light/outputs/phase208/phase208_20261020'
if(-not(Test-Path -LiteralPath $root -PathType Container)){throw 'Phase208 run directory missing'}
if(-not(Test-Path -LiteralPath "$root/DELIVERY_RECEIPT.json")){throw 'Phase208 receipt missing'}
[IO.File]::WriteAllText("$root/RECEIVER_PID",[string]$PID)
function Get-PortableSHA256([string]$path) {
 $stream=[IO.File]::OpenRead($path)
 $hasher=[Security.Cryptography.SHA256]::Create()
 try { return [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
 finally { $stream.Dispose();$hasher.Dispose() }
}
try {
 $receipt=Get-Content -Raw "$root/DELIVERY_RECEIPT.json" | ConvertFrom-Json
 $dest="$root/phase208_delivery.zip"
 if(-not(Test-Path -LiteralPath $dest)) {
  if(-not((Test-Path -LiteralPath "$dest.partial") -and (Get-Item -LiteralPath "$dest.partial").Length -eq $receipt.bytes)) {
  for($attempt=1;$attempt -le 4;$attempt++) {
   scp -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 gzs:/home/scv/shared/phase208_20261020/phase208_delivery.zip "$dest.partial"
   if($LASTEXITCODE -eq 0){break}
   if($attempt -eq 4){throw 'SCP failed after bounded retries'}
   Start-Sleep -Seconds 5
  }
  }
  if((Get-Item -LiteralPath "$dest.partial").Length -ne $receipt.bytes){throw 'archive length mismatch'}
  if((Get-PortableSHA256 "$dest.partial") -ne $receipt.sha256){throw 'archive SHA mismatch'}
  Move-Item -LiteralPath "$dest.partial" -Destination $dest
 }
 & 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B 'D:/Cloud-Adapter-light/src/tools/bounded_phase208_local.py'
 if($LASTEXITCODE -ne 0){throw 'local analysis failed'}
 [IO.File]::WriteAllText("$root/RECEIVER_EXIT_CODE","0`n")
} catch {
 [IO.File]::WriteAllText("$root/RECEIVER_EXIT_CODE","1`n")
 throw
}

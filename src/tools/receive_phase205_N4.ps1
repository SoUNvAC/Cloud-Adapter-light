$ErrorActionPreference='Stop'
$root='D:/Cloud-Adapter-light/outputs/phase205/phase205_20261019'
$dest="$root/N4_transfer"
[IO.Directory]::CreateDirectory($dest) | Out-Null
[IO.File]::WriteAllText("$root/N4_RECEIVER_PID",[string]$PID)
try {
 foreach($entry in @(@('N4_score_recovery/DELIVERY_RECEIPT.json','DELIVERY_RECEIPT.json'),@('N4_score_recovery/N4_scores_delivery.zip','N4_scores_delivery.zip'),@('CALL_LEDGER.jsonl','CALL_LEDGER_FINAL.jsonl'))) {
  $remote=$entry[0];$local=$entry[1]
  for($attempt=1;$attempt -le 4;$attempt++) {
   scp -o ConnectTimeout=15 "gzs:/home/scv/shared/phase205_20261019/$remote" "$dest/$local"
   if($LASTEXITCODE -eq 0){break}
   if($attempt -eq 4){throw 'SCP auxiliary delivery failed'}
   Start-Sleep -Seconds 5
  }
 }
 # Main CPU job extracts the original manifest before any auxiliary verification needs it.
 [IO.File]::WriteAllText("$root/N4_TRANSFER_COMPLETE","1`n")
 [IO.File]::WriteAllText("$root/N4_RECEIVER_EXIT_CODE","0`n")
} catch {
 [IO.File]::WriteAllText("$root/N4_RECEIVER_EXIT_CODE","1`n")
 throw
}

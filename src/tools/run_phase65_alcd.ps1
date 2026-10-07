$ErrorActionPreference = 'Stop'
Set-Location 'D:\Cloud-Adapter-light'
$phase65AlcdLogs = 'outputs/phase65/alcd_download'
New-Item -ItemType Directory -Force -Path $phase65AlcdLogs | Out-Null
Set-Content "$phase65AlcdLogs/PID" -Value $PID
& python -u src/tools/download_phase65_alcd.py --root data/sentinel2_alcd_1460961 >> "$phase65AlcdLogs/console.log" 2>> "$phase65AlcdLogs/error.log"
$phase65AlcdCode = $LASTEXITCODE
if ($phase65AlcdCode -eq 0) {
    & python src/tools/audit_phase65_alcd.py --root data/sentinel2_alcd_1460961 --output "$phase65AlcdLogs/catalogue_audit.json" >> "$phase65AlcdLogs/console.log" 2>> "$phase65AlcdLogs/error.log"
    $phase65AlcdCode = $LASTEXITCODE
}
Set-Content "$phase65AlcdLogs/EXIT_CODE" -Value $phase65AlcdCode
exit $phase65AlcdCode

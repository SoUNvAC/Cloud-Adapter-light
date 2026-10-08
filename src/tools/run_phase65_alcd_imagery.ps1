$ErrorActionPreference = 'Stop'
Set-Location 'D:\Cloud-Adapter-light'
$phase65ImageryLogs = 'outputs/phase65/alcd_imagery'
New-Item -ItemType Directory -Force -Path $phase65ImageryLogs | Out-Null
Set-Content "$phase65ImageryLogs/PID" -Value $PID
& python -u src/tools/acquire_phase65_alcd_imagery.py download --root data/sentinel2_alcd_1460961/imagery >> "$phase65ImageryLogs/console.log" 2>> "$phase65ImageryLogs/error.log"
$phase65ImageryCode = $LASTEXITCODE
Set-Content "$phase65ImageryLogs/EXIT_CODE" -Value $phase65ImageryCode
exit $phase65ImageryCode

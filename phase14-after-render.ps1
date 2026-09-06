[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$RenderOwnerPid,
    [ValidateSet('RTX 3060', 'RTX 4070')]
    [string]$Target = 'RTX 4070',
    [ValidateRange(1, 20)]
    [int]$QuietPolls = 4
)

$quiet = 0
while ($quiet -lt $QuietPolls) {
    $renderChildren = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'ffmpeg.exe' -and $_.ParentProcessId -eq $RenderOwnerPid
    }
    if ($renderChildren) {
        $quiet = 0
    } else {
        $quiet++
    }
    Start-Sleep -Seconds 15
}

$root = $PSScriptRoot
$logDir = Join-Path $root 'logs\shell'
$logPath = Join-Path $logDir 'phase14-autotune.latest.log'
$python = Join-Path $root 'app\env\Scripts\python.exe'
$autotuner = Join-Path $root 'app\tests\phase14_autotune.py'

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
& $python $autotuner --target $Target --start 0 --end 600 `
    --enhancer 'GPEN 256 Pro' --mask-engine 'RealityUX' `
    --stabilization on --force *> $logPath
exit $LASTEXITCODE

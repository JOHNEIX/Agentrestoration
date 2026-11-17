param([switch]$NoBrowser)

$uiUrl = 'http://127.0.0.1:8501'
try {
    $health = Invoke-WebRequest -Uri "$uiUrl/_stcore/health" -UseBasicParsing -TimeoutSec 3
    $alreadyRunning = $health.StatusCode -eq 200 -and ([string]$health.Content).Trim() -eq 'ok'
} catch {
    $alreadyRunning = $false
}

if ($alreadyRunning) {
    Write-Host "The workbench is already running: $uiUrl"
    if (-not $NoBrowser) {
        Start-Process -FilePath $uiUrl | Out-Null
    }
    exit 0
}

$env:PYTHONPATH = (Join-Path $PSScriptRoot 'src') + ';' + $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.cache\pip'
$env:TEMP = Join-Path $PSScriptRoot '.cache\temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Push-Location $PSScriptRoot
try {
    $headless = $NoBrowser.IsPresent.ToString().ToLowerInvariant()
    & '.\.venv\Scripts\python.exe' -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless $headless --browser.gatherUsageStats false
    $uiExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $uiExitCode

param(
    [string]$PythonPath,
    [switch]$NonInteractive,
    [switch]$NoStart,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = (Join-Path $projectRoot 'src') + ';' + $projectRoot
foreach ($entry in @{
    PIP_CACHE_DIR='pip'; TEMP='temp'; TMP='temp'; TORCH_HOME='torch_home';
    HF_HOME='hf_home'; YOLO_CONFIG_DIR='yolo_config_dir'
}.GetEnumerator()) {
    $directory = Join-Path $projectRoot ('.cache\' + $entry.Value)
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    [Environment]::SetEnvironmentVariable($entry.Key, $directory, 'Process')
}

function Invoke-CheckedPython {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python step failed (exit $LASTEXITCODE). Fix the error above and run install.ps1 again." }
}

function Test-PythonVersion {
    param([string]$Executable, [string[]]$Prefix=@())
    try {
        & $Executable @Prefix -c 'import sys,struct; sys.exit(0 if sys.version_info[:2]==(3,12) and struct.calcsize(''P'')==8 else 1)' *> $null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

function Read-ApiKey {
    param([string]$Prompt)
    $secure = Read-Host $Prompt -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer).Trim() }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
}

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if ($CheckOnly) { throw 'No .venv found. Run install.ps1 first.' }
        $executable = $PythonPath
        $prefix = @()
        if ($PythonPath) {
            if (-not (Test-PythonVersion $PythonPath)) { throw 'PythonPath must point to Python 3.12 (64-bit).' }
        } else {
            if ((Get-Command py -ErrorAction SilentlyContinue) -and (Test-PythonVersion 'py' @('-3.12'))) {
                $executable = 'py'; $prefix = @('-3.12')
            } elseif ((Get-Command python -ErrorAction SilentlyContinue) -and (Test-PythonVersion 'python')) {
                $executable = 'python'
            } else {
                throw 'Install Python 3.12 (64-bit), or use: .\install.ps1 -PythonPath C:\path\python.exe'
            }
        }
        Write-Host '[1/5] Creating .venv in the project directory...'
        Invoke-CheckedPython $executable ($prefix + @('-m', 'venv', (Join-Path $projectRoot '.venv')))
    }
    if (-not (Test-PythonVersion $venvPython)) { throw 'Existing .venv must use Python 3.12 (64-bit).' }
    Invoke-CheckedPython $venvPython @('-c', 'import sys,pathlib; assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(''.venv'').resolve(), ''Invalid virtual environment''')

    if (-not $CheckOnly) {
        Write-Host '[2/5] Installing locked dependencies...'
        Invoke-CheckedPython $venvPython @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', 'requirements.lock.txt')
        Write-Host '[3/5] Preparing YOLO11n and three restoration tools...'
        Invoke-CheckedPython $venvPython @('scripts/setup_assets.py')
    }
    Write-Host '[4/5] Checking dependencies, configuration and assets...'
    Invoke-CheckedPython $venvPython @('-m', 'pip', 'check')
    Invoke-CheckedPython $venvPython @('-c', 'import torch,streamlit,ultralytics,einops,httpx; from agentrestoration.core import Settings; from agentrestoration.restoration_tools import RestorationTools; Settings.load(''configs/default.json''); print(''Runtime and configuration OK; torch=''+str(torch.__version__))')
    Invoke-CheckedPython $venvPython @('scripts/setup_assets.py', '--check')

    Write-Host '[5/5] Checking API credentials...'
    $keyCheck = 'import json; from agentrestoration.adapters import credentials; print(json.dumps({name:bool(value.strip()) for name,value in credentials().items()}))'
    $keyStatus = & $venvPython -c $keyCheck
    if ($LASTEXITCODE -ne 0) { throw 'Credential check failed.' }
    $keys = $keyStatus | ConvertFrom-Json
    $apiPath = Join-Path $projectRoot 'api.txt'
    if (-not ($keys.vlm -and $keys.image)) {
        if (-not (Test-Path -LiteralPath $apiPath) -and -not $CheckOnly) {
            if ($NonInteractive) {
                Copy-Item -LiteralPath (Join-Path $projectRoot 'api.example.txt') -Destination $apiPath
            } else {
                $vlmKey = Read-ApiKey 'Vision API key (hidden input)'
                $imageKey = Read-ApiKey 'Image-edit API key (Enter to use the same key)'
                if (-not $imageKey) { $imageKey = $vlmKey }
                if (-not $vlmKey) { throw 'API key is empty. Run again or copy api.example.txt to api.txt and fill it in.' }
                $template = [IO.File]::ReadAllLines((Join-Path $projectRoot 'api.example.txt'), [Text.Encoding]::UTF8)
                [IO.File]::WriteAllLines($apiPath, @($template[0], $vlmKey, $template[2], $imageKey), (New-Object Text.UTF8Encoding($false)))
                Remove-Variable vlmKey, imageKey
            }
        }
        $keyStatus = & $venvPython -c $keyCheck
        if ($LASTEXITCODE -ne 0) { throw 'Credential check failed.' }
        $keys = $keyStatus | ConvertFrom-Json
    }
    if ($keys.vlm -and $keys.image) {
        Write-Host 'API credentials found. No API requests were sent.'
    } else {
        Write-Host "Fill in $apiPath (see api.example.txt), or set VLM_API_KEY and IMAGE_API_KEY."
    }
    Write-Host "Environment ready: $venvPython"
    Write-Host 'API endpoints and model names: configs/default.json'
    Write-Host 'Start the workbench: .\run_ui.ps1'
    if (-not ($CheckOnly -or $NonInteractive -or $NoStart) -and $keys.vlm -and $keys.image) {
        & (Join-Path $projectRoot 'run_ui.ps1') -Restart
    }
} catch {
    Write-Host ('Installation stopped: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
} finally { Pop-Location }

# Project-local CPython bootstrap. Dot-source this file from install.ps1.
function Test-PythonVersion {
    param([string]$Executable, [string[]]$Prefix=@())
    try {
        & $Executable @Prefix -c 'import sys,struct,venv,ensurepip,ssl; sys.exit(0 if sys.version_info[:2]==(3,12) and struct.calcsize(''P'')==8 else 1)' *> $null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

function Receive-PythonFile {
    param([string]$Uri, [string]$Destination)
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $previousProgress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        for ($attempt = 0; $attempt -lt 3; $attempt++) {
            try {
                Invoke-WebRequest -Uri $Uri -OutFile ($Destination + '.part') -UseBasicParsing -TimeoutSec 120
                Move-Item -LiteralPath ($Destination + '.part') -Destination $Destination -Force
                return
            } catch {
                if ($attempt -eq 2) { throw "Python download failed: $Uri. Check your network and rerun install.cmd." }
                Start-Sleep -Seconds 1
            }
        }
    } finally { $ProgressPreference = $previousProgress }
}

function Install-ProjectPython {
    param([Parameter(Mandatory=$true)][string]$ProjectRoot)
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'This project requires 64-bit Windows.' }
    $version = '3.12.9'
    $runtimeRoot = Join-Path $ProjectRoot ".runtime\python-$version"
    $python = Join-Path $runtimeRoot 'tools\python.exe'
    if (Test-PythonVersion $python) {
        Write-Host "Using project Python: $python"
        return $python
    }
    $downloads = Join-Path $ProjectRoot '.cache\downloads'
    New-Item -ItemType Directory -Path $downloads -Force | Out-Null
    $archive = Join-Path $downloads "python.$version.zip"
    $hashPath = $archive + '.sha512'
    $base = "https://api.nuget.org/v3-flatcontainer/python/$version/python.$version.nupkg"
    $cached = $false
    if ((Test-Path -LiteralPath $archive) -and (Test-Path -LiteralPath $hashPath)) {
        $expected = [IO.File]::ReadAllText($hashPath).Trim()
        $actual = [Convert]::ToBase64String([Security.Cryptography.SHA512]::Create().ComputeHash([IO.File]::ReadAllBytes($archive)))
        $cached = $actual -ceq $expected
    }
    if (-not $cached) {
        Write-Host "Downloading project Python $version (64-bit)..."
        $registrationPath = Join-Path $downloads 'python-registration.json'
        $catalogPath = Join-Path $downloads 'python-catalog.json'
        Receive-PythonFile "https://api.nuget.org/v3/registration5-semver1/python/$version.json" $registrationPath
        $registration = Get-Content -LiteralPath $registrationPath -Raw | ConvertFrom-Json
        $catalogUri = [Uri]$registration.catalogEntry
        if ($catalogUri.Scheme -ne 'https' -or $catalogUri.Host -ne 'api.nuget.org') { throw 'Unexpected Python catalog URL.' }
        Receive-PythonFile $catalogUri.AbsoluteUri $catalogPath
        $catalog = Get-Content -LiteralPath $catalogPath -Raw | ConvertFrom-Json
        if ($catalog.packageHashAlgorithm -ne 'SHA512' -or [Convert]::FromBase64String($catalog.packageHash).Length -ne 64) {
            throw 'Invalid Python package checksum metadata.'
        }
        [IO.File]::WriteAllText($hashPath, $catalog.packageHash)
        Receive-PythonFile $base $archive
        $expected = [IO.File]::ReadAllText($hashPath).Trim()
        $actual = [Convert]::ToBase64String([Security.Cryptography.SHA512]::Create().ComputeHash([IO.File]::ReadAllBytes($archive)))
        if ($actual -cne $expected) { throw 'Python package checksum mismatch. Rerun install.cmd to download it again.' }
    }
    Write-Host "Extracting Python into $runtimeRoot ..."
    Expand-Archive -LiteralPath $archive -DestinationPath $runtimeRoot -Force
    if (-not (Test-PythonVersion $python)) { throw 'Downloaded Python failed its version/venv/SSL check.' }
    return $python
}

param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CliArgs)
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.cache\pip'
$env:TEMP = Join-Path $PSScriptRoot '.cache\temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Push-Location $PSScriptRoot
try { & '.\.venv\Scripts\python.exe' -m agentrestoration.cli @CliArgs; $cliExitCode = $LASTEXITCODE }
finally { Pop-Location }
exit $cliExitCode

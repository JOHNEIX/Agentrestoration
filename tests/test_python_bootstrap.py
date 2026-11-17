"""Exercise PowerShell bootstrap with simulated downloads and real venv creation."""
import base64
import hashlib
import shutil
import subprocess
import sys
import zipfile

import pytest

from agentrestoration.core import ROOT


def powershell(script, environment):
    import os
    return subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                          env=dict(os.environ, **environment), capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("mode", ["download", "cache", "installed", "bad_hash", "network_error"])
def test_project_runtime_installation(tmp_path, mode):
    project = tmp_path / "project with spaces"
    project.mkdir()
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("tools/python.exe", "runtime fixture")
    expected = base64.b64encode(hashlib.sha512(archive.read_bytes()).digest()).decode()
    downloads = project / ".cache/downloads"
    downloads.mkdir(parents=True)
    if mode == "cache":
        shutil.copyfile(archive, downloads / "python.3.12.9.zip")
        (downloads / "python.3.12.9.zip.sha512").write_text(expected)
    elif mode == "installed":
        executable = project / ".runtime/python-3.12.9/tools/python.exe"
        executable.parent.mkdir(parents=True)
        executable.write_text("runtime fixture")
    harness = tmp_path / "bootstrap-test.ps1"
    harness.write_text(r'''
$ErrorActionPreference = 'Stop'
. $env:BOOTSTRAP_SCRIPT
# Do not execute any code from the simulated package.
function Test-PythonVersion { param($Executable) return (Test-Path -LiteralPath $Executable) }
$script:requests = 0
function Receive-PythonFile {
    param($Uri, $Destination)
    $script:requests++
    if ($env:BOOTSTRAP_MODE -eq 'network_error') { throw 'simulated network error' }
    if ($Destination.EndsWith('registration.json')) {
        '{"catalogEntry":"https://api.nuget.org/v3/catalog0/fixture.json"}' | Set-Content -LiteralPath $Destination
    } elseif ($Destination.EndsWith('catalog.json')) {
        $hash = $env:BOOTSTRAP_HASH
        if ($env:BOOTSTRAP_MODE -eq 'bad_hash') { $hash = [Convert]::ToBase64String((New-Object byte[] 64)) }
        @{packageHashAlgorithm='SHA512';packageHash=$hash} | ConvertTo-Json | Set-Content -LiteralPath $Destination
    } else { Copy-Item -LiteralPath $env:BOOTSTRAP_ARCHIVE -Destination $Destination }
}
try {
    $executable = Install-ProjectPython $env:BOOTSTRAP_ROOT
    if (-not (Test-Path -LiteralPath $executable)) { throw 'No executable' }
    Write-Output "REQUESTS=$script:requests"
} catch { Write-Output $_.Exception.Message; exit 1 }
''', encoding="ascii")
    result = powershell(harness, dict(BOOTSTRAP_SCRIPT=str(ROOT / "scripts/bootstrap_python.ps1"),
        BOOTSTRAP_MODE=mode, BOOTSTRAP_HASH=expected, BOOTSTRAP_ARCHIVE=str(archive), BOOTSTRAP_ROOT=str(project)))
    executable = project / ".runtime/python-3.12.9/tools/python.exe"
    if mode in ("bad_hash", "network_error"):
        assert result.returncode != 0
        assert not executable.exists()
        assert ("checksum mismatch" if mode == "bad_hash" else "simulated network error") in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert executable.read_text() == "runtime fixture"
        assert f"REQUESTS={3 if mode == 'download' else 0}" in result.stdout


def test_download_retries(tmp_path):
    harness = tmp_path / "download-test.ps1"
    harness.write_text(r'''
$ErrorActionPreference = 'Stop'
. $env:BOOTSTRAP_SCRIPT
$script:attempt = 0
function Start-Sleep {}
function Invoke-WebRequest {
    param($Uri, $OutFile, [switch]$UseBasicParsing, $TimeoutSec)
    $script:attempt++
    if ($script:attempt -lt 3) { throw 'interrupted' }
    [IO.File]::WriteAllText($OutFile, 'complete')
}
$target = Join-Path $env:BOOTSTRAP_ROOT 'download.zip'
Receive-PythonFile 'https://example.test/python' $target
if ($script:attempt -ne 3 -or [IO.File]::ReadAllText($target) -ne 'complete') { exit 1 }
if (Test-Path -LiteralPath ($target + '.part')) { exit 1 }
''', encoding="ascii")
    result = powershell(harness, dict(BOOTSTRAP_SCRIPT=str(ROOT / "scripts/bootstrap_python.ps1"), BOOTSTRAP_ROOT=str(tmp_path)))
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_system_python_creates_real_venv_and_stops_on_dependency_failure(tmp_path):
    project = tmp_path / "new project with spaces"
    (project / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "install.ps1", project / "install.ps1")
    # Simulate a PC without Python; the download adapter supplies the host's
    # interpreter. A real venv is created, while dependency downloads stay offline.
    (project / "scripts/bootstrap_python.ps1").write_text(r'''
. $env:BOOTSTRAP_SCRIPT
function Get-Command { param($Name, $ErrorAction) return $null }
function Install-ProjectPython {
    param($ProjectRoot)
    [IO.File]::WriteAllText((Join-Path $ProjectRoot 'bootstrap-called'), 'yes')
    return $env:BOOTSTRAP_HOST_PYTHON
}
''', encoding="ascii")
    (project / "requirements.lock.txt").write_text("installer-intentional-missing-package==0.0.0\n")
    result = powershell(project / "install.ps1", dict(BOOTSTRAP_SCRIPT=str(ROOT / "scripts/bootstrap_python.ps1"),
                        BOOTSTRAP_HOST_PYTHON=sys.executable, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1"))
    assert result.returncode != 0
    assert (project / "bootstrap-called").exists(), result.stdout + result.stderr
    executable = project / ".venv/Scripts/python.exe"
    assert executable.exists(), result.stdout + result.stderr
    validation = subprocess.run([str(executable), "-c", "import sys,venv,pip; print(sys.prefix)"],
                                capture_output=True, text=True)
    assert validation.returncode == 0
    assert str(project / ".venv") in validation.stdout
    assert "Preparing YOLO" not in result.stdout

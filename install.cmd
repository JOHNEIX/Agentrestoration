@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "installExitCode=%ERRORLEVEL%"
if not "%installExitCode%"=="0" if "%~1"=="" pause
exit /b %installExitCode%

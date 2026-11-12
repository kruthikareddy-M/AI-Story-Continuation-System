# AI Story Continuation System - PowerShell Launcher
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "Starting AI Story Continuation System..." -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Cyan

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    & $VenvPython run_app.py
} else {
    python run_app.py
}

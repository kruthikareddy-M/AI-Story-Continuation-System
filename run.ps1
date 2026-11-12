# AI Story Continuation System - Root PowerShell Launcher
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

if (Test-Path (Join-Path $ScriptDir "story-continuation\run_app.py")) {
    Set-Location (Join-Path $ScriptDir "story-continuation")
}

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "Starting AI Story Continuation System..." -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Cyan

$VenvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    & $VenvPython run_app.py
} else {
    python run_app.py
}

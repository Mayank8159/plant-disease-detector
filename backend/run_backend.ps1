$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$pythonExe = Join-Path $PSScriptRoot ".venv312\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
  Write-Error "Python executable not found: $pythonExe"
}

& $pythonExe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

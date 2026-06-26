$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

$ProjectPython = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw "Project environment is missing. Run .\scripts\setup.ps1 first."
}

& $ProjectPython -m dvc repro

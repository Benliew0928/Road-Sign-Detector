$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

$ProjectPython = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw "Project environment is missing. Run .\scripts\setup.ps1 first."
}

if (-not (Test-Path "apps\web\dist")) {
    Push-Location apps\web
    try {
        npm run build
    } finally {
        Pop-Location
    }
}

& $ProjectPython -m roadsign_assist.cli serve `
    --host 127.0.0.1 `
    --port 8000 `
    --config configs/inference/experimental.yaml

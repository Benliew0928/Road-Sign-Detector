$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

python -m uv sync --python 3.11 --extra dev --extra data --extra training --extra inference --extra ocr

Push-Location apps\web
try {
    npm ci
} finally {
    Pop-Location
}

python -m uv run roadsign-assist doctor

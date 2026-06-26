$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

Push-Location apps\web
try {
    npm run build
} finally {
    Pop-Location
}

Write-Host "Frontend build complete. Windows installer packaging is implemented in P19."


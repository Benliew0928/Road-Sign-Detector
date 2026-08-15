param(
    [string]$Config = "configs/inference/experimental.yaml",
    [int]$Port = 8443,
    [switch]$Public,
    [ValidateSet("cloudflare", "ngrok", "manual")]
    [string]$Provider = "cloudflare",
    [string]$PublicUrl = "",
    [string]$PhoneHost = "",
    [switch]$ListAddresses,
    [switch]$NoBrowser,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

$ProjectPython = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw "Project environment is missing. Run .\scripts\setup.ps1 first."
}

if ($ListAddresses) {
    & "$PSScriptRoot\run_phone.ps1" -ListAddresses
    exit $LASTEXITCODE
}

if (-not $SkipBuild) {
    npm run build --prefix apps\web
}

$DashboardUrl = if ($Public) {
    "http://127.0.0.1:$Port"
} else {
    "https://127.0.0.1:$Port"
}
if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param([string]$Url, [int]$ReadyPort)
        $Deadline = (Get-Date).AddMinutes(2)
        while ((Get-Date) -lt $Deadline) {
            if (Test-NetConnection -ComputerName "127.0.0.1" -Port $ReadyPort -InformationLevel Quiet) {
                Start-Process $Url
                return
            }
            Start-Sleep -Seconds 1
        }
    } -ArgumentList $DashboardUrl, $Port | Out-Null
}

if ($Public) {
    & "$PSScriptRoot\run_public_phone.ps1" `
        -Config $Config `
        -Port $Port `
        -Provider $Provider `
        -PublicUrl $PublicUrl `
        -SkipBuild
    exit $LASTEXITCODE
}

Write-Host "Starting the complete RoadSign Assist website at $DashboardUrl"
Write-Host "The same dashboard includes image, video, laptop camera, phone camera, live wall, OCR, tracking, and offline advisory audio."

& "$PSScriptRoot\run_phone.ps1" `
    -Config $Config `
    -Port $Port `
    -PhoneHost $PhoneHost `
    -SkipBuild

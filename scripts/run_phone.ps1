param(
    [string]$Config = "configs/inference/default.yaml",
    [int]$Port = 8443,
    [string]$PhoneHost = "",
    [switch]$ListAddresses,
    [switch]$RegenerateCert
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

$ProjectPython = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw "Project environment is missing. Run .\scripts\setup.ps1 first."
}

function Get-PhoneAddressCandidates {
    $VirtualPattern = "VirtualBox|VMware|Hyper-V|vEthernet|WSL|Loopback|TAP|TUN|VPN|OpenVPN|ExpressVPN|Wintun|Bluetooth"
    Get-NetIPConfiguration |
        Where-Object {
            $_.IPv4Address -and
            $_.NetAdapter.Status -eq "Up" -and
            $_.IPv4Address.IPAddress -notlike "127.*" -and
            $_.IPv4Address.IPAddress -notlike "169.254*" -and
            $_.NetAdapter.InterfaceDescription -notmatch $VirtualPattern -and
            $_.InterfaceAlias -notmatch $VirtualPattern
        } |
        Sort-Object `
            @{ Expression = { if ($_.IPv4DefaultGateway) { 0 } else { 1 } } }, `
            @{ Expression = { if ($_.InterfaceAlias -match "Wi-Fi|Wireless") { 0 } elseif ($_.InterfaceAlias -match "Ethernet") { 1 } else { 2 } } } |
        ForEach-Object {
            [PSCustomObject]@{
                IPAddress = $_.IPv4Address.IPAddress
                InterfaceAlias = $_.InterfaceAlias
                Description = $_.NetAdapter.InterfaceDescription
                HasGateway = [bool]$_.IPv4DefaultGateway
            }
        }
}

$Candidates = @(Get-PhoneAddressCandidates)

if ($ListAddresses) {
    if ($Candidates.Count -eq 0) {
        Write-Host "No phone-reachable IPv4 candidates found."
    } else {
        $Candidates | Format-Table -AutoSize
    }
    exit 0
}

$IpAddress = $PhoneHost
if (-not $IpAddress -and $Candidates.Count -gt 0) {
    $IpAddress = $Candidates[0].IPAddress
}

if (-not $IpAddress) {
    $IpAddress = "127.0.0.1"
}

$CertPath = Join-Path $PWD "certs\roadsign-local.crt"
$KeyPath = Join-Path $PWD "certs\roadsign-local.key"
$CertHostsPath = Join-Path $PWD "certs\roadsign-local.hosts.txt"
$KnownCertHosts = if (Test-Path -LiteralPath $CertHostsPath) {
    Get-Content -LiteralPath $CertHostsPath
} else {
    @()
}
if (
    $RegenerateCert -or
    -not (Test-Path -LiteralPath $CertPath) -or
    -not (Test-Path -LiteralPath $KeyPath) -or
    ($KnownCertHosts -notcontains $IpAddress)
) {
    & $ProjectPython scripts\create_phone_cert.py --cert $CertPath --key $KeyPath --hosts $IpAddress
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $CertHostsPath) | Out-Null
    @($IpAddress) | Set-Content -LiteralPath $CertHostsPath -Encoding UTF8
}

if (-not (Test-Path "apps\web\dist")) {
    Push-Location apps\web
    try {
        npm run build
    } finally {
        Pop-Location
    }
}

Write-Host "RoadSign Assist phone camera server"
if ($Candidates.Count -gt 0) {
    Write-Host "Selected adapter: $($Candidates[0].InterfaceAlias) - $($Candidates[0].Description)"
}
Write-Host "Laptop URL: https://127.0.0.1:$Port"
Write-Host "Phone URL:  https://$IpAddress`:$Port/phone"
Write-Host "List addresses: .\scripts\run_phone.ps1 -ListAddresses"
Write-Host "Manual IP:       .\scripts\run_phone.ps1 -PhoneHost <your-laptop-wifi-ip> -Config $Config"
Write-Host "If the phone warns about the certificate, install/trust certs\roadsign-local.crt for this local demo."

& $ProjectPython -m roadsign_assist.cli serve `
    --host 0.0.0.0 `
    --port $Port `
    --config $Config `
    --public-host "$IpAddress`:$Port" `
    --ssl-certfile $CertPath `
    --ssl-keyfile $KeyPath

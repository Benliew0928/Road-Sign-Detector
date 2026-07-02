param(
    [string]$Config = "configs/inference/default.yaml",
    [int]$Port = 8443,
    [ValidateSet("cloudflare", "ngrok", "manual")]
    [string]$Provider = "cloudflare",
    [string]$PublicUrl = "",
    [string]$DemoSecret = "",
    [string]$OperatorToken = "",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

$ProjectPython = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw "Project environment is missing. Run .\scripts\setup.ps1 first."
}

function New-RoadSignToken {
    param([int]$Bytes = 32)
    $Buffer = New-Object byte[] $Bytes
    $Generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $Generator.GetBytes($Buffer)
    } finally {
        $Generator.Dispose()
    }
    return [Convert]::ToBase64String($Buffer).TrimEnd("=").Replace("+", "-").Replace("/", "_")
}

function Get-FileText {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return ""
    }
    return Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue
}

function Wait-ForCloudflareUrl {
    param(
        [string]$StdoutPath,
        [string]$StderrPath,
        [int]$TimeoutSeconds = 60
    )
    $Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $Deadline) {
        $Text = "$(Get-FileText $StdoutPath)`n$(Get-FileText $StderrPath)"
        $Match = [regex]::Match($Text, "https://[a-zA-Z0-9-]+\.trycloudflare\.com")
        if ($Match.Success) {
            return $Match.Value
        }
        Start-Sleep -Milliseconds 500
    }
    return ""
}

function Wait-ForNgrokUrl {
    param([int]$TimeoutSeconds = 60)
    $Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $Deadline) {
        try {
            $Response = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 2
            $HttpsTunnel = @($Response.tunnels | Where-Object { $_.public_url -like "https://*" })[0]
            if ($HttpsTunnel.public_url) {
                return $HttpsTunnel.public_url
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return ""
}

if (-not $SkipBuild) {
    npm run build --prefix apps\web
}

if (-not $DemoSecret) {
    $DemoSecret = New-RoadSignToken
}
if (-not $OperatorToken) {
    $OperatorToken = New-RoadSignToken -Bytes 24
}

$TunnelProcess = $null
$LogsDir = Join-Path $PWD "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

try {
    $ResolvedPublicUrl = $PublicUrl.Trim().TrimEnd("/")
    if (-not $ResolvedPublicUrl) {
        if ($Provider -eq "manual") {
            throw "Provide -PublicUrl https://your-public-host for manual tunnel mode."
        }

        if ($Provider -eq "cloudflare") {
            $Cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
            if (-not $Cloudflared) {
                throw "cloudflared is not installed. Install it or use -Provider manual -PublicUrl https://..."
            }
            $StdoutPath = Join-Path $LogsDir "cloudflared-phone.stdout.log"
            $StderrPath = Join-Path $LogsDir "cloudflared-phone.stderr.log"
            Remove-Item -LiteralPath $StdoutPath, $StderrPath -ErrorAction SilentlyContinue
            $TunnelProcess = Start-Process `
                -FilePath $Cloudflared.Source `
                -ArgumentList @("tunnel", "--url", "http://127.0.0.1:$Port", "--no-autoupdate") `
                -RedirectStandardOutput $StdoutPath `
                -RedirectStandardError $StderrPath `
                -WindowStyle Hidden `
                -PassThru
            $ResolvedPublicUrl = Wait-ForCloudflareUrl -StdoutPath $StdoutPath -StderrPath $StderrPath
            if (-not $ResolvedPublicUrl) {
                throw "Cloudflare Tunnel did not return a public URL. Check $StdoutPath and $StderrPath."
            }
        }

        if ($Provider -eq "ngrok") {
            $Ngrok = Get-Command ngrok -ErrorAction SilentlyContinue
            if (-not $Ngrok) {
                throw "ngrok is not installed. Install it or use -Provider manual -PublicUrl https://..."
            }
            $StdoutPath = Join-Path $LogsDir "ngrok-phone.stdout.log"
            $StderrPath = Join-Path $LogsDir "ngrok-phone.stderr.log"
            Remove-Item -LiteralPath $StdoutPath, $StderrPath -ErrorAction SilentlyContinue
            $TunnelProcess = Start-Process `
                -FilePath $Ngrok.Source `
                -ArgumentList @("http", "http://127.0.0.1:$Port", "--log=stdout") `
                -RedirectStandardOutput $StdoutPath `
                -RedirectStandardError $StderrPath `
                -WindowStyle Hidden `
                -PassThru
            $ResolvedPublicUrl = Wait-ForNgrokUrl
            if (-not $ResolvedPublicUrl) {
                throw "ngrok did not return a public HTTPS URL. Check $StdoutPath and $StderrPath."
            }
        }
    }

    if ($ResolvedPublicUrl -notlike "https://*") {
        throw "Public phone mode requires an HTTPS URL so mobile browsers can use the camera."
    }

    $env:ROADSIGN_PUBLIC_BASE_URL = $ResolvedPublicUrl
    $env:ROADSIGN_TUNNEL_MODE = "1"
    $env:ROADSIGN_DEMO_SECRET = $DemoSecret
    $env:ROADSIGN_OPERATOR_TOKEN = $OperatorToken

    $EscapedOperatorToken = [Uri]::EscapeDataString($OperatorToken)
    $OperatorDashboardUrl = "$ResolvedPublicUrl/?operator=$EscapedOperatorToken"
    $OperatorLiveUrl = "$ResolvedPublicUrl/live?operator=$EscapedOperatorToken"

    Write-Host "RoadSign Assist public phone camera server"
    Write-Host "Local dashboard:     http://127.0.0.1:$Port"
    Write-Host "Public viewer URL:   $ResolvedPublicUrl"
    Write-Host "Operator dashboard:  $OperatorDashboardUrl"
    Write-Host "Public live wall:    $OperatorLiveUrl"
    Write-Host "QR source:           Open the local dashboard and use the Phone camera panel."
    Write-Host "Public QR source:    Use the operator dashboard URL, not the viewer URL."
    Write-Host "Privacy:             Keep the public live wall URL private; stop this script to close the tunnel."

    & $ProjectPython -m roadsign_assist.cli serve `
        --host 127.0.0.1 `
        --port $Port `
        --config $Config
} finally {
    if ($TunnelProcess -and -not $TunnelProcess.HasExited) {
        Stop-Process -Id $TunnelProcess.Id -Force
    }
}

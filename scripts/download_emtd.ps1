$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

$destinationRoot = "data\raw\emtd"
$archive = Join-Path $destinationRoot "EMTD.zip"
$downloadUrl = "https://zenodo.org/api/records/1217105/files/EMTD.zip/content"
$expectedMd5 = "2e718023d82dc170ed1ae08c377e08b9"

New-Item -ItemType Directory -Path $destinationRoot -Force | Out-Null

curl.exe --fail --location --continue-at - --output $archive $downloadUrl
if ($LASTEXITCODE -ne 0) {
    throw "EMTD download failed with curl exit code $LASTEXITCODE"
}

$actualMd5 = (Get-FileHash -LiteralPath $archive -Algorithm MD5).Hash.ToLowerInvariant()
if ($actualMd5 -ne $expectedMd5) {
    throw "EMTD checksum mismatch. Expected $expectedMd5, received $actualMd5"
}

Write-Host "Verified $archive"


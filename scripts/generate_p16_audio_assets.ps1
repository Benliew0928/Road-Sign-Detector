param(
    [string]$Output = "apps/web/public/audio/p16/advisory_audio_manifest.json",
    [string]$PublicRoot = "apps/web/public",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

$ProjectPython = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw "Project environment is missing. Run .\scripts\setup.ps1 first."
}

Add-Type -AssemblyName System.Speech

function Select-VoiceName {
    param(
        [System.Speech.Synthesis.SpeechSynthesizer]$Synthesizer,
        [string]$Language
    )
    $Voices = @($Synthesizer.GetInstalledVoices() | Where-Object { $_.Enabled })
    $VoiceInfo = $Voices | ForEach-Object { $_.VoiceInfo }
    if ($Language -eq "zh") {
        $Chinese = @($VoiceInfo | Where-Object { $_.Culture.Name -like "zh-*" })[0]
        if ($Chinese) { return $Chinese.Name }
    }
    if ($Language -eq "ms") {
        $Malay = @($VoiceInfo | Where-Object { $_.Culture.Name -like "ms-*" })[0]
        if ($Malay) { return $Malay.Name }
    }
    $EnglishFemale = @($VoiceInfo | Where-Object { $_.Culture.Name -like "en-*" -and $_.Gender -eq "Female" })[0]
    if ($EnglishFemale) { return $EnglishFemale.Name }
    $English = @($VoiceInfo | Where-Object { $_.Culture.Name -like "en-*" })[0]
    if ($English) { return $English.Name }
    if ($VoiceInfo.Count -gt 0) { return $VoiceInfo[0].Name }
    throw "No installed Windows speech voices are available."
}

& $ProjectPython scripts\build_p16_audio_manifest.py --output $Output --public-root $PublicRoot
if ($LASTEXITCODE -ne 0) {
    throw "Unable to build P16 audio manifest."
}

$ManifestPath = Join-Path $PWD $Output
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$Synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
$AudioFormat = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo `
    22050, `
    ([System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen), `
    ([System.Speech.AudioFormat.AudioChannel]::Mono)
$VoiceNames = @{}
foreach ($Language in $Manifest.languages) {
    $VoiceNames[$Language] = Select-VoiceName -Synthesizer $Synthesizer -Language $Language
}

Write-Host "Generating P16 advisory audio with local Windows voices:"
$VoiceNames.GetEnumerator() | Sort-Object Name | ForEach-Object {
    Write-Host "  $($_.Name): $($_.Value)"
}

$Count = 0
$Total = ($Manifest.phrases.PSObject.Properties | Measure-Object).Count * $Manifest.languages.Count

foreach ($PhraseProperty in $Manifest.phrases.PSObject.Properties) {
    $Phrase = $PhraseProperty.Value
    foreach ($Language in $Manifest.languages) {
        $Asset = $Phrase.assets.$Language
        $Text = [string]$Phrase.text.$Language
        $RelativePath = ([string]$Asset.src).TrimStart("/")
        $OutputPath = Join-Path (Join-Path $PWD $PublicRoot) $RelativePath
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath) | Out-Null
        if ((-not $Force) -and (Test-Path -LiteralPath $OutputPath)) {
            $Count += 1
            continue
        }
        $Synthesizer.SelectVoice([string]$VoiceNames[$Language])
        $Synthesizer.Rate = if ($Language -eq "zh") { -1 } else { 0 }
        $Synthesizer.Volume = 100
        $Synthesizer.SetOutputToWaveFile($OutputPath, $AudioFormat)
        $Synthesizer.Speak($Text)
        $Synthesizer.SetOutputToNull()
        $Count += 1
        if ($Count % 25 -eq 0 -or $Count -eq $Total) {
            Write-Host "  generated/checked $Count / $Total"
        }
    }
}

$VoiceJsonPath = Join-Path $PWD "outputs\p16_voice_names.json"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $VoiceJsonPath) | Out-Null
$VoiceNames | ConvertTo-Json -Compress | Set-Content -LiteralPath $VoiceJsonPath -Encoding UTF8
& $ProjectPython scripts\build_p16_audio_manifest.py `
    --output $Output `
    --public-root $PublicRoot `
    --update-assets `
    --voice-names-file $VoiceJsonPath
if ($LASTEXITCODE -ne 0) {
    throw "Unable to update P16 audio asset metadata."
}

Write-Host "P16 audio assets are ready under $PublicRoot\audio\p16"

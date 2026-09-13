# RoadSign Assist website guide

This is the single operating guide for the RoadSign Assist website. The site is
one FastAPI and React application; image, video, laptop camera, phone camera,
multi-phone live wall, OCR, tracking, and offline advisory audio are views of
the same runtime rather than separate websites.

## Start everything

From `C:\MiniProject`, run:

```powershell
.\scripts\run.ps1
```

The launcher builds the current frontend, creates or refreshes the local HTTPS
certificate when required, starts the complete inference API, and opens
`https://127.0.0.1:8443`. Keep the PowerShell window open while using the site;
press Ctrl+C to stop it.

Run `.\scripts\setup.ps1` once first on a new machine. Restore DVC artifacts
with `.\.venv\Scripts\dvc.exe pull` if the launcher reports a missing model or
audio bundle.

Useful options still use the same launcher:

```powershell
# Do not open the browser automatically
.\scripts\run.ps1 -NoBrowser

# Show candidate LAN addresses
.\scripts\run.ps1 -ListAddresses

# Select a laptop Wi-Fi/hotspot address explicitly
.\scripts\run.ps1 -PhoneHost 192.168.137.1

# Create a temporary trusted public HTTPS tunnel
.\scripts\run.ps1 -Public

# Use ngrok or an already-created HTTPS tunnel
.\scripts\run.ps1 -Public -Provider ngrok
.\scripts\run.ps1 -Public -Provider manual -PublicUrl https://example.test
```

Cloudflare Quick Tunnel is the default public provider and requires
`cloudflared` on PATH. Public mode prints a viewer URL and a private operator
URL. Stop the launcher to close the tunnel. Quick Tunnel URLs and process IDs
are temporary and must never be stored as a "current link" in source control.

## Website functions

The local website uses the **Night Signal** dark interface. There is no account,
login, registration, or cloud-storage workflow. Navigate between **Analyze**,
**Live**, and **Recent**. Page transitions crossfade where supported, with a
short CSS entrance fallback; reduced-motion preferences disable motion.

| Function | How to use it |
| --- | --- |
| Image analysis | Choose **Analyze → Images**, add one or more images, set **Road scene** or **Close-up sign** for each, check orientation and analyze. Click a sign box or finding to inspect it. |
| Batch analysis | Add several images in Images. Review each thumbnail's type and orientation. Results include per-file failures and **Retry failed images**; successful items are retained. Maximum 100 images, 20 MB each. |
| Video analysis | Choose **Analyze → Video**, add a file (up to 250 MB), and use the annotated video and **Explore sign moments**. Findings and warning language follow playback. |
| Laptop camera | Choose **Live → This device**, allow browser permission, and start the camera. Changing source stops this device's camera. |
| Phone camera | Choose **Live → Connected phones → Add phone**, scan the QR, trust `certs\roadsign-local.crt` on the phone when using LAN HTTPS, then start the stream. Rear/front and resolution controls are under **Camera settings**. |
| Multi-phone live wall | Connected phones appear directly in Live. Select a tile to focus it; Escape closes focus. The standalone wall remains available from the pairing panel. Stale frames and delayed analysis are labeled separately. |
| Recent | Reopen up to 24 image results from this browser session. Each retains its image type and runtime description. Reloading clears this history; no account or persistent library is created. |
| OCR and tracking | These run inside the same analysis pipeline when enabled by the selected inference profile. |
| Advisory audio | Choose English, Bahasa Melayu, or Mandarin and unmute. Warnings play from bundled local audio without an online TTS call. |
| Model health | The status area names the detector, classifier, release status, and active profile so an experimental or coverage-blocked model is not presented as clean-final. |

**System details** contains model profiles, bundle identity and diagnostics.
Image batches dispatch sequentially to the appropriate existing single-image
endpoint; the progress count represents completed requests, including failed
items. The video endpoint returns at completion, so video uses indeterminate
processing feedback. The existing batch API remains available to API clients.

The selected classifier is the clean-data EfficientNetV2-S candidate
`clean_b2_effnetv2s_224_e40_b32_s2513`. It scored 437/462 (94.59%) on the
locked test split and 59/84 (70.24%) on the held-out assignment manifest. It is
not a clean-final release while the dataset metadata reports
`coverage_gaps_block_final`; the website keeps that status visible even though
the candidate replaces the older synthetic-assisted classifier.

## Phone and public-camera behavior

The phone sends JPEG frames over the camera WebSocket while model execution
stays on the laptop. The sender limits in-flight frames, adapts JPEG quality to
latency, counts dropped ticks rather than building an unbounded queue, and
returns recognition events to the phone UI. Local mode binds HTTPS on the LAN;
the phone and laptop must be on the same Wi-Fi or hotspot.

Public mode creates signed phone links and an operator token. Public requests
to phone link creation, the live wall, and monitor APIs require that operator
token. Phone and WebSocket links require the signed session access token. The
application retains only the latest live frames in memory and does not record
phone footage to disk.

Troubleshooting:

- Camera permission fails: use the HTTPS URL and trust the generated local
  certificate, or use `-Public` for a browser-trusted tunnel.
- QR points to the wrong adapter: run `.\scripts\run.ps1 -ListAddresses`, then
  restart with `-PhoneHost <address>`.
- Public mode reports `cloudflared is not installed`: install `cloudflared`,
  select `-Provider ngrok`, or pass a trusted manual HTTPS URL.
- A public phone or live wall returns 403: create a fresh QR from the current
  operator dashboard; tokens are intentionally scoped to the current run.
- Live FPS is low: use 640 px phone resolution, improve Wi-Fi, or reduce the
  number of concurrent streams.

## Offline advisory audio

The dashboard first tries the optional AI voice manifest under
`/audio/p16_ai/`, then falls back to the DVC-backed Windows voice pack under
`/audio/p16/`. Audio playback respects the selected language, mute state,
phrase cooldown, and event priority; it never blocks visual inference.

Regenerate and validate the fallback pack with:

```powershell
.\scripts\generate_p16_audio_assets.ps1
.\.venv\Scripts\python.exe -m pytest tests\unit\test_advisory_audio.py -q
```

AI voice generation is an asset-production task, never a live runtime call.
Use `scripts\generate_p16_ai_audio_assets.py --help` for sample, full, resume,
provider, model, voice, timeout, and fallback options. Review a small sample
before generating a full replacement pack, and do not commit API keys.

## Verification

```powershell
.\.venv\Scripts\pytest.exe
npm test --prefix apps\web
npm run build --prefix apps\web
npm run test:e2e --prefix apps\web
```

Automated checks cover the API contract, desktop/mobile layouts, QR flow,
camera routes, audio policy, and core dashboard interactions. Camera permission
and long-duration phone streaming still require a physical-device check.

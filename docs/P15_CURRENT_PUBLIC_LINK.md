# P15 Current Public Cloudflare Link

Last updated: 2026-07-02

## Current Running Demo

This run is currently exposed through a Cloudflare Quick Tunnel:

- Public viewer URL: `https://worker-surprising-cookies-pressing.trycloudflare.com`
- Operator dashboard URL: `https://worker-surprising-cookies-pressing.trycloudflare.com/?operator=-Ya1VzSg5u4vNr_W_Tj3DQtamwvXFGFv`
- Operator live wall URL: `https://worker-surprising-cookies-pressing.trycloudflare.com/live?operator=-Ya1VzSg5u4vNr_W_Tj3DQtamwvXFGFv`

Use the operator dashboard URL when you want to create phone QR links or open
the live camera wall. The plain public viewer URL can load the dashboard, but it
cannot create phone stream links or view all cameras.

## Why The Operator Token Is Needed

In public tunnel mode, RoadSign Assist protects camera-control routes:

- `/api/v1/phone/connection` requires the operator token before it creates a
  signed phone QR link.
- `/live` and the live-wall WebSocket require the operator token before they
  show connected phone camera streams.
- `/phone` requires a signed `session` plus `access` token from the QR link.

This prevents random visitors from opening the live wall or creating camera
stream links.

## Run It Again

From PowerShell:

```powershell
cd C:\MiniProject
.\scripts\run_public_phone.ps1 -Config configs\inference\experimental.yaml -Port 8443
```

The command prints a new public viewer URL, operator dashboard URL, and public
live wall URL. With Cloudflare Quick Tunnel, the public URL usually changes
every time the script restarts.

## Keep It Online

This is not a 24/7 cloud-hosted backend. Cloudflare is only tunneling traffic
to the RoadSign Assist server running on this PC. The link works only while:

- this PC is on and awake;
- the PowerShell runner is still running;
- the Python backend is still running;
- `cloudflared` is still connected;
- the PC has internet access.

For a stable public domain, use a named Cloudflare Tunnel with a domain, and
keep the origin backend running on an always-on PC or cloud server.

## Stop The Current Run

For the current run only:

```powershell
Stop-Process -Id 74640,73128,73324,74748 -Force
```

Process IDs change on every run, so use the IDs printed or listed by
PowerShell for future sessions.

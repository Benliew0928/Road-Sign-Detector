# P15 Phone-Camera Streaming

P15 lets a phone browser act as the wireless camera while the laptop keeps the
FastAPI inference session and model execution local.

## Start The Phone-Camera Server

From `C:\MiniProject`:

```powershell
.\scripts\run_phone.ps1
```

The script:

- builds the web app if needed;
- creates `certs\roadsign-local.crt` and `certs\roadsign-local.key` if missing;
- starts FastAPI on `https://0.0.0.0:8443`;
- prints the LAN phone URL.

Use the normal baseline config by default. To use another profile:

```powershell
.\scripts\run_phone.ps1 -Config configs\inference\experimental.yaml -Port 8443
```

If the phone and laptop are on networks that block local peer-to-peer traffic,
use the public HTTPS tunnel runner instead:

```powershell
.\scripts\run_public_phone.ps1 -Config configs\inference\experimental.yaml -Port 8443
```

See `docs\P15_PUBLIC_TUNNEL.md` for Cloudflare/ngrok/manual tunnel setup,
operator token behavior, and public soak-test steps.

## Connect The Phone

1. Put laptop and phone on the same Wi-Fi or laptop hotspot.
2. Open the laptop dashboard at `https://127.0.0.1:8443`.
3. Select **Phone**.
4. Scan the QR code with the phone.
5. If the phone blocks the camera, install/trust `certs\roadsign-local.crt` for
   this local demo and reopen the QR link.
6. On the phone page, choose rear/front camera and resolution, then tap
   **Start stream**.

## Runtime Behavior

- The phone sends JPEG frames over `/api/v1/ws/camera/{session_id}`.
- Up to two frames are in flight; extra ticks are counted as dropped instead of
  building queue latency.
- JPEG quality adapts between 38% and 78% based on observed round-trip latency.
- The sender targets 60 FPS and falls back toward 30 FPS when latency or drops
  rise.
- WebSocket reconnects are retried up to six times locally, or up to twelve
  times in public tunnel mode.
- Recognition events return to the phone and appear in the current-sign and
  event panels.
- Public tunnel mode signs phone URLs and protects the live wall with an
  operator token.

## Current Verification

Automated checks cover the API connection contract, QR flow, phone route,
desktop layout, and mobile layout. Manual Android/iPhone browser permission and
30-minute offline soak tests still require physical devices.

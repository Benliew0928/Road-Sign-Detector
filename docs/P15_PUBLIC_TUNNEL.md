# P15 Public Phone Tunnel

This is the internet-access path for the P15 phone camera demo. Use it when the
phone and laptop cannot reach each other on local Wi-Fi, such as school Wi-Fi
with client isolation, or when the phone is on mobile data.

The offline LAN/hotspot path is still `scripts\run_phone.ps1`. Public tunnel
mode is optional and should not be required for normal local testing.

## Provider Choice

Primary provider: Cloudflare Quick Tunnel via `cloudflared`.

Reason: it gives a trusted public HTTPS URL without installing a local
certificate on each phone, and it can proxy traffic to the laptop app running on
`127.0.0.1`.

Fallbacks:

- `ngrok`, when an ngrok account and auth token are already configured.
- Manual mode, when another HTTPS tunnel already exists.

Official references:

- Cloudflare Quick Tunnels: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/
- Cloudflare `cloudflared` downloads: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/
- ngrok Agent CLI quickstart: https://ngrok.com/docs/getting-started

## Start A Public Demo

Cloudflare:

```powershell
.\scripts\run_public_phone.ps1 -Config configs\inference\experimental.yaml -Port 8443
```

Manual HTTPS tunnel:

```powershell
.\scripts\run_public_phone.ps1 -Provider manual -PublicUrl https://your-public-host.example -Config configs\inference\experimental.yaml -Port 8443
```

ngrok:

```powershell
.\scripts\run_public_phone.ps1 -Provider ngrok -Config configs\inference\experimental.yaml -Port 8443
```

The script builds `apps\web`, starts the selected tunnel, sets tunnel/security
environment variables, and starts the API on `http://127.0.0.1:8443`.

The terminal prints:

- local dashboard URL;
- public base URL;
- protected public live-wall URL;
- a reminder to keep the operator live-wall URL private.

Open the local dashboard, switch to Phone input, and scan the QR. The QR should
point to the public HTTPS host in tunnel mode.

## Security Behavior

Public tunnel mode sets `ROADSIGN_PUBLIC_BASE_URL` and generates signed access
links:

- `/phone` requires `session` plus signed `access` token on public requests.
- `/api/v1/ws/camera/{session_id}` requires the same signed `access` token on
  public WebSocket connections.
- `/live`, `/api/v1/phone/streams`, and `/api/v1/ws/phone/monitor` require the
  operator token on public requests.
- Local laptop access still works without the public operator token.

Defaults:

- phone access token lifetime: 6 hours;
- operator token lifetime/secret process lifetime: current run only unless
  `ROADSIGN_OPERATOR_TOKEN` is supplied;
- maximum connected phone streams: `ROADSIGN_PHONE_MAX_STREAMS`, default `12`;
- maximum single frame upload: 20 MB.

The app keeps only the latest live frames in memory for dashboard display and
does not record phone footage to disk. The phone page shows a camera sharing
notice before streaming.

## Test Checklist

School Wi-Fi blocked-LAN test:

1. Connect laptop and phone to school Wi-Fi.
2. Confirm the local LAN URL is blocked or unreliable.
3. Start `scripts\run_public_phone.ps1`.
4. Scan the public QR.
5. Start phone stream and confirm the live wall shows that phone.

Mobile-data remote test:

1. Keep laptop on any internet connection.
2. Put phone on mobile data or another network.
3. Scan/copy the public phone URL.
4. Confirm camera permission, phone stream, recognition events, and live wall.

Public 30-minute soak:

1. Start public mode with the intended inference config.
2. Connect one or more phones.
3. Leave streams active for 30 minutes.
4. Record live FPS, AI FPS, reconnect count, latency, and visible errors.
5. Stop the script to close the tunnel.

## Troubleshooting

- `cloudflared is not installed`: install `cloudflared` or use manual/ngrok mode.
- Cloudflare tunnel returns no URL: check `outputs\logs\cloudflared-phone.*.log`.
- Phone gets `403`: scan a fresh QR from the current server run.
- Live wall gets `403`: use the full printed live-wall URL containing
  `?operator=...`.
- Phone camera permission fails: confirm the phone URL is HTTPS.
- QR still shows local IP: confirm `run_public_phone.ps1` is the active server
  and the dashboard was refreshed after startup.
- Very low FPS: reduce phone resolution to 640 px, move closer to the router,
  or reduce connected phone count. Public tunnels add internet latency.

Cloudflare Quick Tunnels are for testing/development. For a production or public
deployment, use a managed tunnel/domain plus stronger identity access controls.

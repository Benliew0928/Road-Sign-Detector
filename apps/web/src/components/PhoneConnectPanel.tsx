import {
  Copy,
  MonitorPlay,
  QrCode,
  RefreshCw,
  ShieldAlert,
  Smartphone,
  Wifi,
} from "lucide-react";
import QRCode from "qrcode";
import { useCallback, useEffect, useState } from "react";

import { getPhoneConnection } from "../api";
import type { PhoneConnectionResponse } from "../types";

interface PhoneConnectPanelProps {
  busy: boolean;
}

function operatorTokenFromUrl(): string | undefined {
  return (
    new URLSearchParams(window.location.search).get("operator") ?? undefined
  );
}

export function PhoneConnectPanel({ busy }: PhoneConnectPanelProps) {
  const [connection, setConnection] = useState<PhoneConnectionResponse | null>(
    null,
  );
  const [qrUrl, setQrUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(false);
  const [operatorToken] = useState(operatorTokenFromUrl);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    setCopied(false);
    try {
      const next = await getPhoneConnection(operatorToken);
      setConnection(next);
      setQrUrl(
        await QRCode.toDataURL(next.phone_url, {
          errorCorrectionLevel: "M",
          margin: 2,
          width: 260,
          color: {
            dark: "#10151b",
            light: "#ffffff",
          },
        }),
      );
    } catch (cause) {
      setConnection(null);
      setQrUrl(null);
      const message =
        cause instanceof Error ? cause.message : "Unable to create phone link.";
      setError(
        message === "Operator access token is required."
          ? "Public QR creation requires the host/operator link. Open the local dashboard on this PC, or use the public link that includes the operator token."
          : message,
      );
    } finally {
      setLoading(false);
    }
  }, [operatorToken]);

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);

  const copyLink = useCallback(async () => {
    if (!connection) return;
    try {
      await navigator.clipboard.writeText(connection.phone_url);
      setCopied(true);
    } catch {
      setError(
        "Clipboard is unavailable. Select and copy the phone link manually.",
      );
    }
  }, [connection]);

  const fallbackLiveWallHref = operatorToken
    ? `/live?operator=${encodeURIComponent(operatorToken)}`
    : null;
  const liveWallHref = connection?.operator_live_url ?? fallbackLiveWallHref;
  const publicMode = connection?.mode === "public_tunnel";

  return (
    <section className="phone-connect" aria-label="Phone camera connection">
      <header>
        <div>
          <span>Phone camera</span>
          <h2>Scan to stream from your phone</h2>
        </div>
        <div className="phone-connect-actions">
          {connection ? (
            <span className={`status-pill ${publicMode ? "online" : ""}`}>
              {publicMode ? "Public tunnel" : "Local"}
            </span>
          ) : null}
          {liveWallHref ? (
            <a
              className="live-stream-button"
              href={liveWallHref}
              title="Open host live camera wall"
            >
              <MonitorPlay size={18} />
              <span>Open host live camera wall</span>
            </a>
          ) : (
            <button
              className="live-stream-button"
              type="button"
              disabled
              title="Use the operator dashboard link printed by the public runner"
            >
              <MonitorPlay size={18} />
              <span className="sr-only">Open host live camera wall</span>
            </button>
          )}
          <button
            className="icon-button"
            onClick={() => void refresh()}
            disabled={loading || busy}
          >
            <RefreshCw size={17} />
            <span className="sr-only">Refresh phone QR</span>
          </button>
        </div>
      </header>

      <div className="phone-connect-body">
        <div className="qr-frame">
          {qrUrl ? (
            <img src={qrUrl} alt="Phone camera connection QR code" />
          ) : (
            <div className="qr-placeholder">
              <QrCode size={42} aria-hidden="true" />
              <span>{loading ? "Generating QR" : "QR unavailable"}</span>
            </div>
          )}
        </div>

        <div className="phone-connect-steps">
          <div>
            <Wifi size={18} aria-hidden="true" />
            <span>
              {publicMode
                ? "Public HTTPS tunnel is active for phones outside local Wi-Fi."
                : "Connect laptop and phone to the same Wi-Fi or hotspot."}
            </span>
          </div>
          <div>
            <Smartphone size={18} aria-hidden="true" />
            <span>
              Scan the QR, allow camera access, then tap Start stream.
            </span>
          </div>
          <div>
            <ShieldAlert size={18} aria-hidden="true" />
            <span>
              Phone cameras need a trusted HTTPS connection. Connection help is
              below.
            </span>
          </div>
        </div>
      </div>

      {connection ? (
        <div className="phone-link-box">
          <span>
            {copied
              ? "Copied phone link"
              : publicMode && connection.public_base_url
                ? `${connection.public_base_url} phone link ready`
                : connection.phone_url}
          </span>
          <button
            className="icon-button"
            onClick={() => void copyLink()}
            title="Copy phone link"
          >
            <Copy size={16} />
            <span className="sr-only">Copy phone link</span>
          </button>
        </div>
      ) : null}

      {connection && !connection.https ? (
        <div className="phone-connect-warning">
          This server is using HTTP. Desktop testing works, but most phones
          require HTTPS before camera permission is available.
        </div>
      ) : null}

      {connection?.candidate_urls.length ? (
        <details className="phone-candidates">
          <summary>Connection help & network addresses</summary>
          <p>
            Use the same Wi-Fi or hotspot. If camera permission is blocked,
            trust the local certificate on your phone, then reopen this link.
          </p>
          {connection.candidate_urls.map((url) => (
            <span key={url}>{url}</span>
          ))}
        </details>
      ) : null}

      {error ? (
        <div className="error-banner" role="alert">
          {error}
        </div>
      ) : null}
    </section>
  );
}

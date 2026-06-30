import {
  Camera,
  CameraOff,
  Gauge,
  Radio,
  RotateCcw,
  ShieldCheck,
  Smartphone,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";

import {
  type PhoneFacingMode,
  type PhoneCameraStatus,
  usePhoneCameraStream,
} from "./hooks/usePhoneCameraStream";
import type { SignEvent } from "./types";

function sessionFromUrl(): string {
  return new URLSearchParams(window.location.search).get("session") ?? "";
}

function accessTokenFromUrl(): string {
  return new URLSearchParams(window.location.search).get("access") ?? "";
}

function createDeviceId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `phone-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function deviceIdForSession(sessionId: string): string {
  const storageKey = `roadsign-phone-device:${sessionId || "missing-session"}`;
  try {
    const existing = sessionStorage.getItem(storageKey);
    if (existing) return existing;
    const next = createDeviceId();
    sessionStorage.setItem(storageKey, next);
    return next;
  } catch {
    return createDeviceId();
  }
}

function choosePrimaryEvent(events: SignEvent[]): SignEvent | null {
  if (!events.length) return null;
  return [...events].sort((first, second) => {
    if (first.stable !== second.stable) return first.stable ? -1 : 1;
    return second.confidence - first.confidence;
  })[0];
}

function statusLabel(status: PhoneCameraStatus): string {
  if (status === "requesting") return "Requesting camera";
  if (status === "connecting") return "Connecting";
  if (status === "live") return "Streaming";
  if (status === "error") return "Needs attention";
  return "Ready";
}

export default function PhoneCameraApp() {
  const [sessionId] = useState(sessionFromUrl);
  const [deviceId] = useState(() => deviceIdForSession(sessionFromUrl()));
  const [accessToken] = useState(accessTokenFromUrl);
  const [facingMode, setFacingMode] = useState<PhoneFacingMode>("environment");
  const [maxWidth, setMaxWidth] = useState(960);
  const previewRef = useRef<HTMLElement>(null);
  const [mediaBox, setMediaBox] = useState<CSSProperties | null>(null);
  const { videoRef, status, error, result, events, stats, start, stop } = usePhoneCameraStream({
    sessionId,
    deviceId,
    accessToken,
    publicMode: Boolean(accessToken),
    facingMode,
    maxWidth,
  });
  const latestEvent = useMemo(() => choosePrimaryEvent(result?.events ?? []), [result]);
  const historyEvent = useMemo(() => choosePrimaryEvent(events), [events]);
  const primaryEvent = latestEvent ?? historyEvent;
  const live = status === "live";
  const busy = status === "requesting" || status === "connecting";

  useEffect(() => {
    const preview = previewRef.current;
    if (!preview || !result) {
      setMediaBox(null);
      return;
    }

    const updateBox = () => {
      const width = preview.clientWidth;
      const height = preview.clientHeight;
      if (!width || !height) return;
      const resultAspect = result.width / result.height;
      const previewAspect = width / height;
      if (previewAspect > resultAspect) {
        const fittedWidth = height * resultAspect;
        setMediaBox({
          left: (width - fittedWidth) / 2,
          top: 0,
          width: fittedWidth,
          height,
        });
      } else {
        const fittedHeight = width / resultAspect;
        setMediaBox({
          left: 0,
          top: (height - fittedHeight) / 2,
          width,
          height: fittedHeight,
        });
      }
    };

    updateBox();
    const observer = new ResizeObserver(updateBox);
    observer.observe(preview);
    return () => observer.disconnect();
  }, [result]);

  return (
    <main className="phone-shell">
      <header className="phone-topbar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={22} aria-hidden="true" />
          </div>
          <div>
            <h1>RoadSign Assist</h1>
            <span>Phone camera link</span>
          </div>
        </div>
        <span className={`status-pill ${live ? "online" : status === "error" ? "offline" : ""}`}>
          {live ? <Wifi size={15} /> : <WifiOff size={15} />}
          {statusLabel(status)}
        </span>
      </header>

      <section ref={previewRef} className="phone-preview" aria-label="Phone camera preview">
        {sessionId ? (
          <div className="phone-video-space" style={mediaBox ?? { inset: 0 }}>
            <video ref={videoRef} muted playsInline className="phone-video" />
            {result ? (
              <div className="phone-overlay-layer" aria-label={`${result.events.length} detected signs`}>
                {result.events.map((event) => (
                  <div
                    className={`phone-detection-box severity-${event.severity}`}
                    key={`${event.frame_id}-${event.track_id}`}
                    style={{
                      left: `${(event.bbox.x1 / result.width) * 100}%`,
                      top: `${(event.bbox.y1 / result.height) * 100}%`,
                      width: `${((event.bbox.x2 - event.bbox.x1) / result.width) * 100}%`,
                      height: `${((event.bbox.y2 - event.bbox.y1) / result.height) * 100}%`,
                    }}
                  >
                    <span>
                      #{event.track_id} {event.meaning.en} {Math.round(event.confidence * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : (
          <div className="phone-missing-session">
            <Smartphone size={34} aria-hidden="true" />
            <strong>Scan a fresh QR code</strong>
            <span>This phone link is missing its camera session.</span>
          </div>
        )}
        <div className="phone-preview-hud">
          <span>{live ? "Live uplink" : "Camera idle"}</span>
          <strong>{stats.latencyMs === null ? "-" : `${stats.latencyMs} ms`}</strong>
        </div>
      </section>

      {error ? (
        <div className="error-banner" role="alert">
          {error}
        </div>
      ) : null}

      <section className="phone-consent" aria-label="Camera sharing notice">
        <ShieldCheck size={17} aria-hidden="true" />
        <span>
          Camera frames stream live to the laptop operator for demo analysis. RoadSign Assist keeps
          the latest frames in memory and does not record footage.
        </span>
      </section>

      <section className="phone-controls" aria-label="Phone camera controls">
        <label>
          <span>Camera</span>
          <select
            value={facingMode}
            onChange={(event) => setFacingMode(event.target.value as PhoneFacingMode)}
            disabled={live || busy}
          >
            <option value="environment">Rear camera</option>
            <option value="user">Front camera</option>
          </select>
        </label>
        <label>
          <span>Resolution</span>
          <select
            value={maxWidth}
            onChange={(event) => setMaxWidth(Number(event.target.value))}
            disabled={live || busy}
          >
            <option value={640}>640 px</option>
            <option value={960}>960 px</option>
            <option value={1280}>1280 px</option>
          </select>
        </label>
        <div className="phone-command-row">
          {live || busy ? (
            <button className="primary-command danger" onClick={stop}>
              <CameraOff size={18} />
              Stop stream
            </button>
          ) : (
            <button className="primary-command" onClick={() => void start()} disabled={!sessionId}>
              <Camera size={18} />
              Start stream
            </button>
          )}
          <button className="icon-button" onClick={() => void start()} disabled={!sessionId || busy}>
            <RotateCcw size={17} />
            <span className="sr-only">Restart stream</span>
          </button>
        </div>
      </section>

      <section className="phone-stats" aria-label="Phone stream metrics">
        <div>
          <Radio size={15} />
          <span>Live FPS</span>
          <strong>{stats.sendFps}</strong>
        </div>
        <div>
          <Gauge size={15} />
          <span>Target</span>
          <strong>{stats.targetFps}</strong>
        </div>
        <div>
          <Radio size={15} />
          <span>Sent</span>
          <strong>{stats.framesSent}</strong>
        </div>
        <div>
          <Radio size={15} />
          <span>Acked</span>
          <strong>{stats.framesAcked}</strong>
        </div>
        <div>
          <Gauge size={15} />
          <span>Quality</span>
          <strong>{Math.round(stats.jpegQuality * 100)}%</strong>
        </div>
        <div>
          <Gauge size={15} />
          <span>Dropped</span>
          <strong>{stats.framesDropped}</strong>
        </div>
      </section>

      <section className="phone-current-sign">
        <span className="eyebrow">Current sign</span>
        {primaryEvent ? (
          <>
            <div className="phone-sign-heading">
              <h2>{primaryEvent.meaning.en}</h2>
              <strong>{Math.round(primaryEvent.confidence * 100)}%</strong>
            </div>
            <p>{primaryEvent.action.code.replaceAll("_", " ")}</p>
          </>
        ) : (
          <>
            <h2>No stable sign</h2>
            <p>Keep the road sign centered for a few frames.</p>
          </>
        )}
      </section>

      <section className="phone-events" aria-label="Phone recognition events">
        <header className="section-heading">
          <h2>Events</h2>
          <span>{events.length}</span>
        </header>
        {events.length ? (
          events.slice(0, 8).map((event, index) => (
            <article className="phone-event" key={`${event.track_id}-${event.frame_id}-${index}`}>
              <span className={`event-marker severity-${event.severity}`} />
              <div>
                <strong>{event.meaning.en}</strong>
                <span>#{event.track_id} - {event.action.code.replaceAll("_", " ")}</span>
              </div>
            </article>
          ))
        ) : (
          <div className="timeline-empty">
            <Radio size={18} aria-hidden="true" />
            <span>No recognition events yet</span>
          </div>
        )}
      </section>
    </main>
  );
}

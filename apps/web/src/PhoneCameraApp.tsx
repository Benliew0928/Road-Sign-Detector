import { useLiveDisplay } from "./hooks/useLiveDisplay";
import { useEncounters } from "./hooks/useEncounters";
import { useAdvisoryAudio } from "./hooks/useAdvisoryAudio";

import {
  Camera,
  CameraOff,
  RotateCcw,
  Settings2,
  Volume2,
  VolumeX,
  Radio,
} from "lucide-react";
import { useState } from "react";
import { advisoryInstruction, semanticSignName } from "./advisoryDisplay";
import {
  type PhoneFacingMode,
  type PhoneCameraStatus,
  usePhoneCameraStream,
} from "./hooks/usePhoneCameraStream";
import type { DisplayLanguage } from "./types";
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
function statusLabel(status: PhoneCameraStatus): string {
  if (status === "requesting") return "Requesting camera";
  if (status === "connecting") return "Connecting";
  if (status === "live") return "Streaming";
  if (status === "error") return "Needs attention";
  return "Ready";
}
export default function PhoneCameraApp() {
  const [language, setLanguage] = useState<DisplayLanguage>("en");
  const [sessionId] = useState(sessionFromUrl);
  const [deviceId] = useState(() => deviceIdForSession(sessionFromUrl()));
  const [accessToken] = useState(accessTokenFromUrl);
  const [facingMode, setFacingMode] = useState<PhoneFacingMode>("environment");
  const [maxWidth, setMaxWidth] = useState(4096);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { videoRef, status, error, result, stats, start, stop } =
    usePhoneCameraStream({
      sessionId,
      deviceId,
      accessToken,
      publicMode: Boolean(accessToken),
      facingMode,
      maxWidth,
    });
  const live = status === "live";
  const findings = useEncounters(
    `${sessionId}:${deviceId}:${status}`,
    result,
    live,
  );

  const [muted, setMuted] = useState(false);
  const audio = useAdvisoryAudio({
    encounters: findings,
    source: findings.source,
    language,
    muted,
    enabled: live,
  });
  const busy = status === "requesting" || status === "connecting";
  const displayed = useLiveDisplay(findings);
  const rawPrimary = result?.events
    .filter((event) => event.semantic_sign_id !== "unknown_sign")
    .sort((left, right) => right.confidence - left.confidence)[0] ?? null;
  // Boxes are drawn from the current raw frame. Keep the information card in
  // sync with them even before an encounter has met the stability gate.
  const primary = live ? (displayed?.event ?? rawPrimary) : null;
  const primaryLastSeen = Boolean(
    displayed?.event === primary && displayed?.lastSeen,
  );
  return (
    <main className="phone-immersive" data-live={live}>
      <section className="phone-live-stage" aria-label="Phone camera preview">
        <video
          ref={videoRef}
          muted
          playsInline
          autoPlay
          className="phone-live-video"
        />
        {live && result && result.events.length > 0 && (
          <svg
            className="phone-live-overlay"
            viewBox={`0 0 ${result.width} ${result.height}`}
            preserveAspectRatio="xMidYMid slice"
            aria-label={`${result.events.length} detected signs`}
          >
            {result.events.map((event, index) => {
              const width = Math.max(1, event.bbox.x2 - event.bbox.x1);
              const height = Math.max(1, event.bbox.y2 - event.bbox.y1);
              return (
                <g
                  className={`phone-live-detection severity-${event.severity}`}
                  key={`${event.frame_id}-${event.track_id}-${index}`}
                >
                  <rect
                    x={event.bbox.x1}
                    y={event.bbox.y1}
                    width={width}
                    height={height}
                    vectorEffect="non-scaling-stroke"
                  />
                  <text
                    x={event.bbox.x1}
                    y={Math.max(18, event.bbox.y1 - 7)}
                    vectorEffect="non-scaling-stroke"
                  >
                    {semanticSignName(event, language)}
                  </text>
                </g>
              );
            })}
          </svg>
        )}
        <div className="phone-live-shade" />
      </section>
      <header className="phone-live-header">
        <span className="phone-live-brand">
          <Radio size={18} /> RoadSign Assist
        </span>
        <span
          className={`phone-live-status ${live ? "live" : ""}`}
          role="status"
        >
          <i />
          {statusLabel(status)}
        </span>
      </header>
      {!live && (
        <div className="phone-live-welcome">
          <div className="phone-camera-orbit">
            <Camera size={44} />
          </div>
          <span>YOUR LIVE PERSPECTIVE</span>
          <h1>
            {sessionId
              ? busy
                ? "Connecting your view"
                : "Point. See. Understand."
              : "Connect your camera"}
          </h1>
          <p>
            {sessionId
              ? "Share your camera with the connected computer."
              : "Scan the QR code on your computer to start."}
          </p>
        </div>
      )}
      {(error || audio.error) && (
        <div className="phone-live-error" role="alert">
          {error || audio.error}
          {audio.blocked && (
            <button onClick={audio.enable}>Enable audio</button>
          )}
        </div>
      )}
      {primary && (
        <aside className="phone-live-guidance" aria-label="Current sign">
          <span className="phone-guidance-mark">
            <Radio size={22} />
          </span>
          <div>
            <small>{primaryLastSeen ? "RECENTLY SEEN" : "IN VIEW"}</small>
            <h2>{semanticSignName(primary, language)}</h2>
            <p>{advisoryInstruction(primary, language)}</p>
          </div>
        </aside>
      )}
      <div className="phone-live-actions">
        <button
          className="phone-round-action"
          aria-label="Camera settings"
          aria-expanded={settingsOpen}
          onClick={() => setSettingsOpen((value) => !value)}
        >
          <Settings2 size={22} />
        </button>
        {live || busy ? (
          <button className="phone-stream-action stop" onClick={stop}>
            <CameraOff size={20} />
            Stop stream
          </button>
        ) : (
          <button
            className="phone-stream-action"
            onClick={() => void start()}
            disabled={!sessionId}
          >
            <Camera size={20} />
            Start stream
          </button>
        )}
        <button
          className="phone-round-action"
          aria-label={muted ? "Unmute guidance" : "Mute guidance"}
          onClick={() => setMuted((value) => !value)}
        >
          {muted ? <VolumeX size={22} /> : <Volume2 size={22} />}
        </button>
      </div>
      <span className="phone-live-caption">
        {live ? "Streaming to your computer" : "Live camera · No recording"}
      </span>
      {settingsOpen && (
        <section
          className="phone-live-settings"
          aria-label="Camera settings panel"
        >
          <header>
            <h2>Camera settings</h2>
            <button
              onClick={() => setSettingsOpen(false)}
              aria-label="Close camera settings"
            >
              Done
            </button>
          </header>
          <label>
            Camera
            <select
              aria-label="Camera"
              value={facingMode}
              onChange={(event) =>
                setFacingMode(event.target.value as PhoneFacingMode)
              }
              disabled={live || busy}
            >
              <option value="environment">Rear camera</option>
              <option value="user">Front camera</option>
            </select>
          </label>
          <label>
            Resolution
            <select
              aria-label="Resolution"
              value={maxWidth}
              onChange={(event) => setMaxWidth(Number(event.target.value))}
              disabled={live || busy}
            >
              {[640, 960, 1280, 1920, 4096].map((size) => (
                <option key={size} value={size}>
                  {size === 4096 ? "Device maximum" : `${size} px`}
                </option>
              ))}
            </select>
          </label>
          <label>
            Guidance language
            <select
              aria-label="Results language"
              value={language}
              onChange={(event) =>
                setLanguage(event.target.value as DisplayLanguage)
              }
            >
              <option value="en">English</option>
              <option value="ms">Bahasa Melayu</option>
              <option value="zh">中文</option>
            </select>
          </label>
          {live && (
            <small>Stop streaming to change the camera or resolution.</small>
          )}
          <div className="phone-settings-footer">
            <span>
              {stats.latencyMs === null
                ? "Not connected"
                : `${stats.latencyMs} ms · ${stats.ackFps} fps`}
            </span>
            <button onClick={() => void start()} disabled={!sessionId || busy}>
              <RotateCcw size={15} />
              Restart stream
            </button>
          </div>
        </section>
      )}
    </main>
  );
}

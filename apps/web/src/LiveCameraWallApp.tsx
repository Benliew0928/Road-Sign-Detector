import { useEncounters } from "./hooks/useEncounters";
import { useAdvisoryAudio } from "./hooks/useAdvisoryAudio";
import { EncounterPanel } from "./components/EncounterPanel";
import {
  ChevronLeft,
  ChevronRight,
  MonitorPlay,
  RefreshCw,
  Route,
  Wifi,
  WifiOff,
  X,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { semanticSignName } from "./advisoryDisplay";
import { usePhoneMonitor } from "./hooks/usePhoneMonitor";
import type { DisplayLanguage, PhoneStreamSnapshot } from "./types";
interface GridLayout {
  columns: number;
  rows: number;
}
function calculateGrid(
  count: number,
  width: number,
  height: number,
): GridLayout {
  if (count <= 1) return { columns: 1, rows: 1 };
  if (width <= 0 || height <= 0) {
    const columns = Math.ceil(Math.sqrt(count));
    return { columns, rows: Math.ceil(count / columns) };
  }
  const targetAspect = 16 / 9;
  let best: GridLayout = { columns: count, rows: 1 };
  let bestScore = Number.NEGATIVE_INFINITY;
  for (let columns = 1; columns <= count; columns += 1) {
    const rows = Math.ceil(count / columns);
    const tileWidth = width / columns;
    const tileHeight = height / rows;
    const fittedWidth = Math.min(tileWidth, tileHeight * targetAspect);
    const fittedHeight = Math.min(tileHeight, tileWidth / targetAspect);
    const emptySlots = columns * rows - count;
    const aspectPenalty = Math.abs(tileWidth / tileHeight - targetAspect) * 150;
    const score =
      fittedWidth * fittedHeight - emptySlots * 1200 - aspectPenalty;
    if (score > bestScore) {
      bestScore = score;
      best = { columns, rows };
    }
  }
  return best;
}
function formatAge(updatedAt: number, now: number): string {
  const seconds = Math.max(0, Math.round(now / 1000 - updatedAt));
  if (seconds < 2) return "now";
  if (seconds < 60) return `${seconds}s ago`;
  return `${Math.floor(seconds / 60)}m ago`;
}
function useGridLayout(count: number) {
  const gridRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const grid = gridRef.current;
    if (!grid) return;
    const update = () => {
      const rect = grid.getBoundingClientRect();
      setSize({ width: rect.width, height: rect.height });
    };
    update();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(update);
    observer.observe(grid);
    return () => observer.disconnect();
  }, []);
  return {
    gridRef,
    layout: useMemo(
      () => calculateGrid(count, size.width, size.height),
      [count, size.height, size.width],
    ),
  };
}
function useMediaBox(stream: PhoneStreamSnapshot) {
  const frameRef = useRef<HTMLDivElement>(null);
  const [mediaBox, setMediaBox] = useState<CSSProperties | null>(null);
  const frameWidth = stream.result?.width ?? stream.width;
  const frameHeight = stream.result?.height ?? stream.height;
  useEffect(() => {
    const frame = frameRef.current;
    if (!frame || !frameWidth || !frameHeight) {
      setMediaBox(null);
      return;
    }
    const updateBox = () => {
      const width = frame.clientWidth;
      const height = frame.clientHeight;
      if (!width || !height) return;
      const sourceAspect = frameWidth / frameHeight;
      const frameAspect = width / height;
      if (frameAspect > sourceAspect) {
        const fittedWidth = height * sourceAspect;
        setMediaBox({
          left: (width - fittedWidth) / 2,
          top: 0,
          width: fittedWidth,
          height,
        });
      } else {
        const fittedHeight = width / sourceAspect;
        setMediaBox({
          left: 0,
          top: (height - fittedHeight) / 2,
          width,
          height: fittedHeight,
        });
      }
    };
    updateBox();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(updateBox);
    observer.observe(frame);
    return () => observer.disconnect();
  }, [frameHeight, frameWidth]);
  return { frameRef, mediaBox };
}
interface LiveCameraTileProps {
  language?: DisplayLanguage;
  stream: PhoneStreamSnapshot;
  now: number;
  zoomed?: boolean;
  onOpen?: () => void;
}
function LiveCameraTile({
  stream,
  now,
  zoomed = false,
  onOpen,
  language = "en",
}: LiveCameraTileProps) {
  const { frameRef, mediaBox } = useMediaBox(stream);
  const result = stream.result;
  const stale = now / 1000 - stream.updated_at > 5;
  const aligned = stream.inference_frame_seq === stream.frame_seq;
  const events = result?.events ?? [];
  const findings = useEncounters(stream.stream_id, result, !stale);
  const primaryEvent = findings.primary?.event;
  const imageSrc = stream.jpeg_base64
    ? `data:image/jpeg;base64,${stream.jpeg_base64}`
    : null;
  const content = (
    <>
      <div ref={frameRef} className="live-camera-frame">
        {imageSrc ? (
          <div className="live-media-space" style={mediaBox ?? { inset: 0 }}>
            <img src={imageSrc} alt={`${stream.label} live camera frame`} />
            {result && aligned && !stale ? (
              <div
                className="live-overlay-layer"
                aria-label={`${events.length} detected signs`}
              >
                {events.map((event, index) => (
                  <div
                    className={`live-detection-box severity-${event.severity}`}
                    key={`${event.frame_id}-${event.track_id}-${index}`}
                    style={{
                      left: `${(event.bbox.x1 / result.width) * 100}%`,
                      top: `${(event.bbox.y1 / result.height) * 100}%`,
                      width: `${((event.bbox.x2 - event.bbox.x1) / result.width) * 100}%`,
                      height: `${((event.bbox.y2 - event.bbox.y1) / result.height) * 100}%`,
                    }}
                  >
                    <span>{semanticSignName(event, language)}</span>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : (
          <div className="live-camera-empty">
            <MonitorPlay size={34} aria-hidden="true" />
            <span>Waiting for frames</span>
          </div>
        )}
      </div>
      <span className={`feed-state ${stale ? "stale" : ""}`}>
        {stale
          ? `Last frame · ${formatAge(stream.updated_at, now)}`
          : !imageSrc
            ? "Waiting for frames"
            : "Receiving frames"}
        {!aligned && result ? " · Analysis catching up" : ""}
      </span>
      <footer className="live-camera-meta">
        <div>
          <strong>{stream.label}</strong>
          <span>
            {primaryEvent
              ? `${!aligned || stale ? "Last finding: " : ""}${semanticSignName(primaryEvent, language)}`
              : "No confirmed sign"}
          </span>
        </div>
        <div>
          <strong>{Math.round(stream.live_fps)} FPS</strong>
          <span>
            AI {stream.inference_fps.toFixed(1)}
            {stream.inference_pending ? " pending" : ""}
          </span>
        </div>
        <div>
          <strong>{events.length}</strong>
          <span>{formatAge(stream.updated_at, now)}</span>
        </div>
      </footer>
    </>
  );
  if (zoomed) {
    return <article className="live-camera-tile zoomed">{content}</article>;
  }
  return (
    <button
      className="live-camera-tile"
      type="button"
      onClick={onOpen}
      aria-label={`Zoom ${stream.label}`}
    >
      {content}
    </button>
  );
}
function statusText(status: string): string {
  if (status === "live") return "Monitor connected";
  if (status === "reconnecting") return "Reconnecting";
  if (status === "error") return "Monitor offline";
  return "Connecting";
}
function dashboardHrefFromOperator(): string {
  const operatorToken = new URLSearchParams(window.location.search).get(
    "operator",
  );
  return operatorToken
    ? `/?operator=${encodeURIComponent(operatorToken)}`
    : "/";
}
export default function LiveCameraWallApp({
  embedded = false,
  language = "en",
  muted: parentMuted,
  onMutedChange,
}: {
  muted?: boolean;
  onMutedChange?: (muted: boolean) => void;
  embedded?: boolean;
  language?: DisplayLanguage;
}) {
  const { streams, status, error, refresh } = usePhoneMonitor();
  const { gridRef, layout } = useGridLayout(streams.length);
  const [now, setNow] = useState(() => Date.now());
  const [zoomStreamId, setZoomStreamId] = useState<string | null>(null);
  const dashboardHref = useMemo(() => dashboardHrefFromOperator(), []);
  const zoomIndex = streams.findIndex(
    (stream) => stream.stream_id === zoomStreamId,
  );
  const zoomedStream = zoomIndex >= 0 ? streams[zoomIndex] : null;
  const [selectedStreamId, setSelectedStreamId] = useState<string | null>(null);
  const selectedStream =
    zoomedStream ??
    streams.find((s) => s.stream_id === selectedStreamId) ??
    streams[0];
  const [muted, setMuted] = useState(false);
  const selectedLive = Boolean(
    selectedStream &&
      now / 1000 - selectedStream.updated_at < 5 &&
      status === "live",
  );
  const findings = useEncounters(
    selectedStream?.stream_id ?? "monitor",
    selectedStream?.result ?? null,
    selectedLive,
  );
  const audio = useAdvisoryAudio({
    encounters: findings,
    source: findings.source,
    language,
    muted: parentMuted ?? muted,
    enabled: selectedLive,
  });
  const dialogRef = useRef<HTMLElement>(null);
  const dialogOpen = Boolean(zoomedStream);
  useEffect(() => {
    if (!dialogOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    const dialog = dialogRef.current;
    dialog?.querySelector<HTMLButtonElement>("button")?.focus();
    const trap = (event: KeyboardEvent) => {
      if (event.key !== "Tab" || !dialog) return;
      const buttons = Array.from(
        dialog.querySelectorAll<HTMLButtonElement>("button"),
      );
      const first = buttons[0];
      const last = buttons[buttons.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", trap);
    return () => {
      document.removeEventListener("keydown", trap);
      previous?.focus();
    };
  }, [dialogOpen]);
  const moveZoom = useCallback(
    (direction: -1 | 1) => {
      if (!streams.length) return;
      const currentIndex = zoomIndex >= 0 ? zoomIndex : 0;
      const nextIndex =
        (currentIndex + direction + streams.length) % streams.length;
      setZoomStreamId(streams[nextIndex].stream_id);
      setSelectedStreamId(streams[nextIndex].stream_id);
    },
    [streams, zoomIndex],
  );
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!zoomedStream) return;
      if (event.key === "Escape") {
        setZoomStreamId(null);
        return;
      }
      if (event.key === "ArrowLeft" || event.key.toLowerCase() === "a") {
        moveZoom(-1);
      }
      if (event.key === "ArrowRight" || event.key.toLowerCase() === "d") {
        moveZoom(1);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [moveZoom, zoomedStream]);
  return (
    <section className={`live-wall-shell ${embedded ? "embedded-wall" : ""}`}>
      <header className="live-wall-topbar">
        {!embedded && (
          <div className="brand">
            <div className="brand-mark">
              <Route size={23} aria-hidden="true" />
            </div>
            <div>
              <h1>RoadSign Assist</h1>
              <span>Host live camera wall</span>
            </div>
          </div>
        )}
        <div className="live-wall-actions">
          <button
            className="quiet-button"
            onClick={() =>
              onMutedChange
                ? onMutedChange(!(parentMuted ?? muted))
                : setMuted((value) => !value)
            }
          >
            {(parentMuted ?? muted) ? "Unmute guidance" : "Mute guidance"}
          </button>
          {audio.blocked && (
            <button onClick={audio.enable}>Enable audio</button>
          )}
          <span className="subtle-note">
            Guidance: {selectedStream?.label ?? "No feed"}
          </span>
          <span
            className={`status-pill ${status === "live" ? "online" : "offline"}`}
          >
            {status === "live" ? <Wifi size={15} /> : <WifiOff size={15} />}
            {statusText(status)}
          </span>
          <span className="status-pill">
            <MonitorPlay size={15} />
            {streams.length} {streams.length === 1 ? "device" : "devices"}
          </span>
          <button
            className="icon-button"
            onClick={() => void refresh()}
            title="Refresh cameras"
          >
            <RefreshCw size={17} />
            <span className="sr-only">Refresh cameras</span>
          </button>
          {!embedded && (
            <a className="live-back-button" href={dashboardHref}>
              <ChevronLeft size={17} />
              Dashboard
            </a>
          )}
        </div>
      </header>
      {selectedStream && (
        <EncounterPanel
          state={findings}
          raw={selectedStream.result}
          language={language}
        />
      )}
      {audio.error && !audio.blocked && <p role="status">{audio.error}</p>}
      {error ? (
        <div className="live-wall-error" role="alert">
          {error}
        </div>
      ) : null}
      <section
        ref={gridRef}
        className="live-camera-grid"
        style={{
          gridTemplateColumns: `repeat(${layout.columns}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${layout.rows}, minmax(0, 1fr))`,
        }}
        aria-label="Connected phone camera live footage"
      >
        {streams.length ? (
          streams.map((stream) => (
            <LiveCameraTile
              key={stream.stream_id}
              stream={stream}
              language={language}
              now={now}
              onOpen={() => {
                setZoomStreamId(stream.stream_id);
                setSelectedStreamId(stream.stream_id);
              }}
            />
          ))
        ) : (
          <div className="live-wall-empty">
            <MonitorPlay size={42} aria-hidden="true" />
            <strong>No phones streaming</strong>
            <span>
              {embedded
                ? "Choose Add phone above, scan the QR and start your camera."
                : "Open the dashboard, choose Live → Add phone, then start each stream."}
            </span>
          </div>
        )}
      </section>
      {zoomedStream ? (
        <section
          ref={dialogRef}
          className="live-zoom-layer"
          role="dialog"
          aria-modal="true"
          aria-label="Camera zoom view"
        >
          <button
            className="live-zoom-close"
            type="button"
            onClick={() => setZoomStreamId(null)}
            title="Close zoom"
          >
            <X size={22} />
            <span className="sr-only">Close zoom</span>
          </button>
          <button
            className="live-zoom-nav previous"
            type="button"
            onClick={() => moveZoom(-1)}
            title="Previous camera"
          >
            <ChevronLeft size={26} />
            <span className="sr-only">Previous camera</span>
          </button>
          <LiveCameraTile
            stream={zoomedStream}
            now={now}
            zoomed
            language={language}
          />
          <button
            className="live-zoom-nav next"
            type="button"
            onClick={() => moveZoom(1)}
            title="Next camera"
          >
            <ChevronRight size={26} />
            <span className="sr-only">Next camera</span>
          </button>
        </section>
      ) : null}
    </section>
  );
}

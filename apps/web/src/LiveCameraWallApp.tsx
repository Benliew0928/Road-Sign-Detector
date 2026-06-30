import { ChevronLeft, MonitorPlay, RefreshCw, ShieldCheck, Wifi, WifiOff, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";

import arrowAsset from "./assets/arrow.png";
import { usePhoneMonitor } from "./hooks/usePhoneMonitor";
import type { PhoneStreamSnapshot, SignEvent } from "./types";

interface GridLayout {
  columns: number;
  rows: number;
}

function choosePrimaryEvent(events: SignEvent[]): SignEvent | null {
  if (!events.length) return null;
  return [...events].sort((first, second) => {
    if (first.stable !== second.stable) return first.stable ? -1 : 1;
    return second.confidence - first.confidence;
  })[0];
}

function calculateGrid(count: number, width: number, height: number): GridLayout {
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
    const score = fittedWidth * fittedHeight - emptySlots * 1200 - aspectPenalty;
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
    layout: useMemo(() => calculateGrid(count, size.width, size.height), [
      count,
      size.height,
      size.width,
    ]),
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
  stream: PhoneStreamSnapshot;
  now: number;
  zoomed?: boolean;
  onOpen?: () => void;
}

function LiveCameraTile({ stream, now, zoomed = false, onOpen }: LiveCameraTileProps) {
  const { frameRef, mediaBox } = useMediaBox(stream);
  const result = stream.result;
  const events = result?.events ?? [];
  const primaryEvent = choosePrimaryEvent(events);
  const imageSrc = stream.jpeg_base64 ? `data:image/jpeg;base64,${stream.jpeg_base64}` : null;
  const content = (
    <>
      <div ref={frameRef} className="live-camera-frame">
        {imageSrc ? (
          <div className="live-media-space" style={mediaBox ?? { inset: 0 }}>
            <img src={imageSrc} alt={`${stream.label} live camera frame`} />
            {result ? (
              <div className="live-overlay-layer" aria-label={`${events.length} detected signs`}>
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
                    <span>{event.meaning.en}</span>
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
      <footer className="live-camera-meta">
        <div>
          <strong>{stream.label}</strong>
          <span>{primaryEvent?.meaning.en ?? "No sign detected"}</span>
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

export default function LiveCameraWallApp() {
  const { streams, status, error, refresh } = usePhoneMonitor();
  const { gridRef, layout } = useGridLayout(streams.length);
  const [now, setNow] = useState(() => Date.now());
  const [zoomStreamId, setZoomStreamId] = useState<string | null>(null);
  const zoomIndex = streams.findIndex((stream) => stream.stream_id === zoomStreamId);
  const zoomedStream = zoomIndex >= 0 ? streams[zoomIndex] : null;

  const moveZoom = useCallback(
    (direction: -1 | 1) => {
      if (!streams.length) return;
      const currentIndex = zoomIndex >= 0 ? zoomIndex : 0;
      const nextIndex = (currentIndex + direction + streams.length) % streams.length;
      setZoomStreamId(streams[nextIndex].stream_id);
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
    <main className="live-wall-shell">
      <header className="live-wall-topbar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={23} aria-hidden="true" />
          </div>
          <div>
            <h1>RoadSign Assist</h1>
            <span>Host live camera wall</span>
          </div>
        </div>
        <div className="live-wall-actions">
          <span className={`status-pill ${status === "live" ? "online" : "offline"}`}>
            {status === "live" ? <Wifi size={15} /> : <WifiOff size={15} />}
            {statusText(status)}
          </span>
          <span className="status-pill">
            <MonitorPlay size={15} />
            {streams.length} {streams.length === 1 ? "device" : "devices"}
          </span>
          <button className="icon-button" onClick={() => void refresh()} title="Refresh cameras">
            <RefreshCw size={17} />
            <span className="sr-only">Refresh cameras</span>
          </button>
          <a className="live-back-button" href="/">
            <ChevronLeft size={17} />
            Dashboard
          </a>
        </div>
      </header>

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
              now={now}
              onOpen={() => setZoomStreamId(stream.stream_id)}
            />
          ))
        ) : (
          <div className="live-wall-empty">
            <MonitorPlay size={42} aria-hidden="true" />
            <strong>No phones streaming</strong>
            <span>Open the QR page, scan with one or more phones, then start each stream.</span>
          </div>
        )}
      </section>

      {zoomedStream ? (
        <section className="live-zoom-layer" role="dialog" aria-modal="true" aria-label="Camera zoom view">
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
            <img src={arrowAsset} alt="" />
            <span className="sr-only">Previous camera</span>
          </button>
          <LiveCameraTile stream={zoomedStream} now={now} zoomed />
          <button
            className="live-zoom-nav next"
            type="button"
            onClick={() => moveZoom(1)}
            title="Next camera"
          >
            <img src={arrowAsset} alt="" />
            <span className="sr-only">Next camera</span>
          </button>
        </section>
      ) : null}
    </main>
  );
}

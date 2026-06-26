import {
  Activity,
  Camera,
  CameraOff,
  Cpu,
  Files,
  Film,
  Gauge,
  ImagePlus,
  Languages,
  Maximize2,
  Minimize2,
  Radio,
  RotateCcw,
  ShieldCheck,
  Upload,
  Volume2,
  VolumeX,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getHealth, inferBatch, inferImage, inferVideo } from "./api";
import { BatchResults, type BatchDisplayItem } from "./components/BatchResults";
import { EventTimeline } from "./components/EventTimeline";
import { SignPanel } from "./components/SignPanel";
import { VideoSurface } from "./components/VideoSurface";
import { VideoResults } from "./components/VideoResults";
import { useCameraStream } from "./hooks/useCameraStream";
import type {
  DisplayLanguage,
  FrameResult,
  HealthResponse,
  SignEvent,
  SourceMode,
  VideoInferenceResponse,
} from "./types";

function choosePrimaryEvent(result: FrameResult | null): SignEvent | null {
  if (!result?.events.length) return null;
  return [...result.events].sort((a, b) => {
    if (a.stable !== b.stable) return a.stable ? -1 : 1;
    return b.confidence - a.confidence;
  })[0];
}

function pipelineLabel(mode: FrameResult["mode"] | null): string {
  if (mode === "deep") return "Semantic AI pipeline";
  if (mode === "baseline") return "Classical baseline";
  if (mode === "auto") return "Automatic fallback pipeline";
  return "Checking pipeline";
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [sourceMode, setSourceMode] = useState<SourceMode>("image");
  const [language, setLanguage] = useState<DisplayLanguage>("en");
  const [result, setResult] = useState<FrameResult | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoSummary, setVideoSummary] = useState<VideoInferenceResponse | null>(null);
  const [batchItems, setBatchItems] = useState<BatchDisplayItem[]>([]);
  const [history, setHistory] = useState<SignEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [muted, setMuted] = useState(false);
  const [presenterMode, setPresenterMode] = useState(false);
  const [operationError, setOperationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const batchInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const batchItemsRef = useRef<BatchDisplayItem[]>([]);

  const handleResult = useCallback((next: FrameResult) => {
    setResult(next);
    const notable = next.events.filter((event) => event.stable || next.mode === "baseline");
    if (notable.length) {
      setHistory((current) => [...notable.reverse(), ...current].slice(0, 40));
    }
  }, []);

  const camera = useCameraStream(handleResult);

  const refreshHealth = useCallback(async () => {
    const controller = new AbortController();
    try {
      setHealthError(null);
      setHealth(await getHealth(controller.signal));
    } catch (cause) {
      setHealthError(cause instanceof Error ? cause.message : "Backend unavailable");
    }
    return () => controller.abort();
  }, []);

  useEffect(() => {
    let active = true;
    void getHealth()
      .then((response) => {
        if (active) setHealth(response);
      })
      .catch((cause: unknown) => {
        if (active) {
          setHealthError(cause instanceof Error ? cause.message : "Backend unavailable");
        }
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  useEffect(() => {
    batchItemsRef.current = batchItems;
  }, [batchItems]);

  useEffect(
    () => () => {
      batchItemsRef.current.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    },
    [],
  );

  const switchMode = useCallback(
    (mode: SourceMode) => {
      if (mode !== "camera") camera.stop();
      setSourceMode(mode);
      setResult(null);
      setOperationError(null);
    },
    [camera],
  );

  const handleImage = useCallback(
    async (file: File) => {
      camera.stop();
      setSourceMode("image");
      setBusy(true);
      setOperationError(null);
      const nextUrl = URL.createObjectURL(file);
      setImageUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return nextUrl;
      });
      try {
        const response = await inferImage(file);
        handleResult(response.result);
      } catch (cause) {
        setOperationError(cause instanceof Error ? cause.message : "Image analysis failed.");
      } finally {
        setBusy(false);
      }
    },
    [camera, handleResult],
  );

  const handleBatch = useCallback(
    async (files: File[]) => {
      camera.stop();
      setSourceMode("batch");
      setBusy(true);
      setResult(null);
      setOperationError(null);
      const selected = files.slice(0, 100);
      const pending: BatchDisplayItem[] = selected.map((file) => ({
        filename: file.name,
        previewUrl: URL.createObjectURL(file),
      }));
      batchItemsRef.current.forEach((item) => URL.revokeObjectURL(item.previewUrl));
      setBatchItems(pending);
      try {
        const response = await inferBatch(selected);
        setBatchItems((current) =>
          current.map((item, index) => ({
            ...item,
            result: response.results[index]?.result,
            error: response.results[index]?.error,
          })),
        );
        const firstResult = response.results.find((item) => item.result)?.result ?? null;
        if (firstResult) handleResult(firstResult);
      } catch (cause) {
        setOperationError(cause instanceof Error ? cause.message : "Batch analysis failed.");
      } finally {
        setBusy(false);
      }
    },
    [camera, handleResult],
  );

  const handleVideo = useCallback(
    async (file: File) => {
      camera.stop();
      setSourceMode("video");
      setBusy(true);
      setResult(null);
      setVideoSummary(null);
      setOperationError(null);
      const nextUrl = URL.createObjectURL(file);
      setVideoUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return nextUrl;
      });
      try {
        setVideoSummary(await inferVideo(file));
      } catch (cause) {
        setOperationError(cause instanceof Error ? cause.message : "Video analysis failed.");
      } finally {
        setBusy(false);
      }
    },
    [camera],
  );

  const primaryEvent = useMemo(() => choosePrimaryEvent(result), [result]);
  const modelWarnings = health?.models.warnings ?? [];
  const backendOnline = health?.status === "ok" && !healthError;
  const activeMode = result?.mode ?? health?.models.mode ?? null;
  const runtimeLabel =
    primaryEvent?.device ??
    health?.models.detector_device ??
    health?.models.classifier_providers?.[0] ??
    activeMode ??
    "—";
  const classifierRuntime = health?.models.classifier_providers?.join(", ") ?? "not loaded";

  return (
    <main className={`app-shell ${presenterMode ? "presenter-mode" : ""}`}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={23} aria-hidden="true" />
          </div>
          <div>
            <h1>RoadSign Assist</h1>
            <span>Malaysian ADAS vision</span>
          </div>
        </div>

        <div className="system-summary">
          <span className={`status-pill ${backendOnline ? "online" : "offline"}`}>
            {backendOnline ? <Wifi size={15} /> : <WifiOff size={15} />}
            {backendOnline ? "System ready" : "Backend offline"}
          </span>
          <span className="status-pill">
            <Cpu size={15} />
            {health?.models.mode ?? "checking"}
          </span>
          <button className="icon-button" onClick={() => void refreshHealth()} title="Refresh status">
            <RotateCcw size={17} />
            <span className="sr-only">Refresh status</span>
          </button>
          <button
            className="icon-button"
            onClick={() => setPresenterMode((current) => !current)}
            title={presenterMode ? "Exit presenter mode" : "Presenter mode"}
            aria-pressed={presenterMode}
          >
            {presenterMode ? <Minimize2 size={17} /> : <Maximize2 size={17} />}
            <span className="sr-only">
              {presenterMode ? "Exit presenter mode" : "Presenter mode"}
            </span>
          </button>
        </div>
      </header>

      <div className="workspace">
        <aside className="control-rail">
          <section>
            <span className="rail-label">Input source</span>
            <div className="segmented-control">
              <button
                className={sourceMode === "camera" ? "active" : ""}
                onClick={() => switchMode("camera")}
              >
                <Camera size={17} />
                Camera
              </button>
              <button
                className={sourceMode === "image" ? "active" : ""}
                onClick={() => switchMode("image")}
              >
                <ImagePlus size={17} />
                Image
              </button>
              <button
                className={sourceMode === "batch" ? "active" : ""}
                onClick={() => switchMode("batch")}
              >
                <Files size={17} />
                Batch
              </button>
              <button
                className={sourceMode === "video" ? "active" : ""}
                onClick={() => switchMode("video")}
              >
                <Film size={17} />
                Video
              </button>
            </div>
          </section>

          {sourceMode === "camera" ? (
            <section className="source-actions">
              {camera.status === "live" ? (
                <button className="primary-command danger" onClick={camera.stop}>
                  <CameraOff size={18} />
                  Stop camera
                </button>
              ) : (
                <button
                  className="primary-command"
                  onClick={() => void camera.start()}
                  disabled={!backendOnline || camera.status === "connecting"}
                >
                  <Radio size={18} />
                  {camera.status === "connecting" ? "Connecting" : "Start camera"}
                </button>
              )}
            </section>
          ) : sourceMode === "image" ? (
            <section className="source-actions">
              <input
                ref={fileInputRef}
                className="sr-only"
                type="file"
                accept="image/png,image/jpeg,image/webp,image/bmp"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void handleImage(file);
                  event.target.value = "";
                }}
              />
              <button
                className="primary-command"
                onClick={() => fileInputRef.current?.click()}
                disabled={!backendOnline || busy}
              >
                <Upload size={18} />
                {busy ? "Analyzing" : "Choose image"}
              </button>
            </section>
          ) : sourceMode === "batch" ? (
            <section className="source-actions">
              <input
                ref={batchInputRef}
                className="sr-only"
                type="file"
                multiple
                accept="image/png,image/jpeg,image/webp,image/bmp"
                onChange={(event) => {
                  const files = Array.from(event.target.files ?? []);
                  if (files.length) void handleBatch(files);
                  event.target.value = "";
                }}
              />
              <button
                className="primary-command"
                onClick={() => batchInputRef.current?.click()}
                disabled={!backendOnline || busy}
              >
                <Upload size={18} />
                {busy ? "Analyzing batch" : "Choose images"}
              </button>
            </section>
          ) : (
            <section className="source-actions">
              <input
                ref={videoInputRef}
                className="sr-only"
                type="file"
                accept="video/mp4,video/webm,video/quicktime,video/x-msvideo"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void handleVideo(file);
                  event.target.value = "";
                }}
              />
              <button
                className="primary-command"
                onClick={() => videoInputRef.current?.click()}
                disabled={!backendOnline || busy}
              >
                <Upload size={18} />
                {busy ? "Analyzing video" : "Choose video"}
              </button>
            </section>
          )}

          <section>
            <span className="rail-label">Warning language</span>
            <div className="language-row">
              <div className="language-control">
              <Languages size={17} aria-hidden="true" />
              <select
                value={language}
                onChange={(event) => setLanguage(event.target.value as DisplayLanguage)}
                aria-label="Warning language"
              >
                <option value="en">English</option>
                <option value="ms">Bahasa Melayu</option>
                <option value="zh">中文</option>
              </select>
              </div>
              <button
                className="icon-button"
                onClick={() => setMuted((current) => !current)}
                title={muted ? "Enable warnings" : "Mute warnings"}
                aria-pressed={muted}
              >
                {muted ? <VolumeX size={17} /> : <Volume2 size={17} />}
                <span className="sr-only">{muted ? "Enable warnings" : "Mute warnings"}</span>
              </button>
            </div>
          </section>

          <section className="metrics-stack">
            <span className="rail-label">Live metrics</span>
            <div className="metric-row">
              <Activity size={16} />
              <span>Latency</span>
              <strong>{result ? `${Math.round(result.latency_ms)} ms` : "—"}</strong>
            </div>
            <div className="metric-row">
              <Gauge size={16} />
              <span>FPS</span>
              <strong>
                {result && result.latency_ms > 0 ? (1000 / result.latency_ms).toFixed(1) : "—"}
              </strong>
            </div>
            <div className="metric-row">
              <Radio size={16} />
              <span>Signs</span>
              <strong>{result?.events.length ?? 0}</strong>
            </div>
            <div className="metric-row">
              <Cpu size={16} />
              <span>Runtime</span>
              <strong>{runtimeLabel}</strong>
            </div>
            <div className="metric-row">
              <Cpu size={16} />
              <span>Classifier</span>
              <strong title={classifierRuntime}>{classifierRuntime}</strong>
            </div>
          </section>

          {modelWarnings.length ? (
            <section className="model-warning">
              <strong>Development mode</strong>
              {modelWarnings.map((warning) => (
                <span key={warning}>{warning}</span>
              ))}
            </section>
          ) : null}
        </aside>

        <section className="primary-work">
          {sourceMode === "batch" ? (
            <BatchResults items={batchItems} busy={busy} />
          ) : sourceMode === "video" ? (
            <VideoResults videoUrl={videoUrl} summary={videoSummary} busy={busy} />
          ) : (
            <VideoSurface
              mode={sourceMode}
              videoRef={camera.videoRef}
              imageUrl={imageUrl}
              result={result}
            />
          )}
          {(operationError || camera.error || healthError) && (
            <div className="error-banner" role="alert">
              {operationError || camera.error || healthError}
            </div>
          )}
          <div className="work-footer">
            <span>
              <span className={`status-dot ${camera.status === "live" ? "live" : ""}`} />
              {sourceMode === "camera"
                ? camera.status
                : busy
                  ? "processing"
                  : sourceMode}
            </span>
            <span>
              {pipelineLabel(activeMode)}
            </span>
            <span>Frame {result?.frame_id ?? "—"}</span>
          </div>
        </section>

        <aside className="insight-rail">
          <SignPanel event={primaryEvent} language={language} />
          <section className="vehicle-panel">
            <header className="section-heading">
              <h2>Vehicle state</h2>
              <span className="simulator-badge">SIM</span>
            </header>
            <div className="speed-readout">
              <strong>{primaryEvent?.action.target_speed_kmh ?? 50}</strong>
              <span>km/h target</span>
            </div>
            <div className="vehicle-action">
              <span>Advisory</span>
              <strong>
                {primaryEvent?.action.code.replaceAll("_", " ") ?? "MONITOR ROAD"}
              </strong>
            </div>
          </section>
          <EventTimeline events={history} language={language} />
        </aside>
      </div>
    </main>
  );
}

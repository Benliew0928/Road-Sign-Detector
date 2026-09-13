import type { VideoProgress } from "./api";
import { AnalysisLoader } from "./components/AnalysisLoader";
import {
  ArrowUpRight,
  Camera,
  CameraOff,
  ChevronRight,
  Film,
  History,
  ImagePlus,
  Languages,
  Maximize2,
  Minimize2,
  Radio,
  RotateCcw,
  RotateCw,
  Route,
  ScanLine,
  Smartphone,
  Upload,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { getHealth, inferCloseUpImage, inferImage, inferVideo } from "./api";
import { semanticSignName } from "./advisoryDisplay";
import {
  bakeImageRotation,
  rotateQuarterTurn,
  type QuarterTurn,
} from "./imageOrientation";
import { ImageAnalysisWorkspace } from "./components/ImageAnalysisWorkspace";
import { BatchResults, type BatchDisplayItem } from "./components/BatchResults";
import { PhoneConnectPanel } from "./components/PhoneConnectPanel";
import { MovableDock } from "./components/MovableDock";
import { SystemStatus } from "./components/SystemStatus";
import { VideoResults } from "./components/VideoResults";
import { VideoSurface } from "./components/VideoSurface";
import { EncounterPanel } from "./components/EncounterPanel";
import { useEncounters } from "./hooks/useEncounters";
import { EventTimeline } from "./components/EventTimeline";
import LiveCameraWallApp from "./LiveCameraWallApp";
import { useCameraStream } from "./hooks/useCameraStream";
import { useAdvisoryAudio } from "./hooks/useAdvisoryAudio";
import type {
  DisplayLanguage,
  FrameResult,
  HealthResponse,
  VideoInferenceResponse,
} from "./types";
type ImageType = "road_scene" | "close_up";
type Activity = "analyze" | "live" | "recent";
interface AnalysisItem extends BatchDisplayItem {
  id: string;
  file: File;
  imageType: ImageType;
  rotation: QuarterTurn;
  runtimeLabel?: string;
  detectorRuntime?: string;
  classifierRuntime?: string;
  modelWarnings?: string[];
}
const imageAccept = "image/png,image/jpeg,image/webp,image/bmp";
function typeLabel(type: ImageType) {
  return type === "close_up" ? "Close-up sign" : "Road scene";
}
function transition(action: () => void) {
  if (
    typeof document.startViewTransition === "function" &&
    !window.matchMedia("(prefers-reduced-motion: reduce)").matches
  ) {
    document.startViewTransition(() => flushSync(action));
  } else action();
}
export default function App() {
  const [activity, setActivity] = useState<Activity>("analyze");
  const [mediaType, setMediaType] = useState<"image" | "video">("image");
  const [liveSource, setLiveSource] = useState<"camera" | "phone">("phone");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [operationError, setOperationError] = useState<string | null>(null);
  const [imageType, setImageType] = useState<ImageType>("road_scene");
  const [language, setLanguage] = useState<DisplayLanguage>("en");
  const [muted, setMuted] = useState(false);
  const [presenterMode, setPresenterMode] = useState(false);
  const [items, setItems] = useState<AnalysisItem[]>([]);
  const [recent, setRecent] = useState<AnalysisItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [reviewIndex, setReviewIndex] = useState(0);
  const [preparing, setPreparing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [videoProgress,setVideoProgress] = useState<VideoProgress | null>(null);
  const [progress, setProgress] = useState(0);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoSummary, setVideoSummary] =
    useState<VideoInferenceResponse | null>(null);
  const [videoFailed, setVideoFailed] = useState(false);
  const [cameraResult, setCameraResult] = useState<FrameResult | null>(null);
  const [showPairing, setShowPairing] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const videoInput = useRef<HTMLInputElement>(null);
  const objectUrls = useRef(new Set<string>());
  const mounted = useRef(true);
  const running = useRef(false);
  const rememberUrl = (file: File) => {
    const url = URL.createObjectURL(file);
    objectUrls.current.add(url);
    return url;
  };
  const handleCameraResult = useCallback((next: FrameResult) => {
    setCameraResult(next);
  }, []);
  const camera = useCameraStream(handleCameraResult);
  const selected = items.find((item) => item.id === selectedId) ?? null;
  const audioResult =
    activity === "live" && liveSource === "camera"
      ? cameraResult
      : activity === "analyze" && mediaType === "image"
        ? (selected?.result ?? null)
        : null;
  const cameraFindings = useEncounters(
    `webcam:${camera.status}`,
    cameraResult,
    camera.status === "live",
  );
  const audio = useAdvisoryAudio({
    result: audioResult,
    language,
    muted,
    source: activity === "live" ? cameraFindings.source : `image:${selectedId}`,
    encounters:
      activity === "live" && liveSource === "camera"
        ? cameraFindings
        : undefined,
    enabled:
      activity === "live"
        ? liveSource === "camera" && camera.status === "live"
        : mediaType === "image",
  });
  const online = health?.status === "ok" && !healthError;
  const modelWarnings = health?.models.warnings ?? [];
  const publicHost =
    !/^(localhost$|127\.|192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\.|\[::1\]$|.*\.local$)/i.test(
      window.location.hostname,
    );
  const refreshHealth = useCallback(async () => {
    try {
      const next = await getHealth();
      if (mounted.current) {
        setHealth(next);
        setHealthError(null);
      }
    } catch (error) {
      if (mounted.current)
        setHealthError(
          error instanceof Error ? error.message : "Local service unavailable",
        );
    }
  }, []);
  useEffect(() => {
    mounted.current = true;
    void getHealth()
      .then((next) => {
        if (mounted.current) {
          setHealth(next);
          setHealthError(null);
        }
      })
      .catch((error: unknown) => {
        if (mounted.current)
          setHealthError(
            error instanceof Error
              ? error.message
              : "Local service unavailable",
          );
      });
    return () => {
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    const urls = objectUrls.current;
    return () => {
      urls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);
  useEffect(
    () => () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    },
    [videoUrl],
  );
  function navigate(next: Activity) {
    if (busy) return;
    if (next !== "live") {
      camera.stop();
      setCameraResult(null);
    }
    transition(() => {
      setActivity(next);
      setOperationError(null);
    });
  }
  function uploadImages(files: File[]) {
    if (running.current || !files.length) return;
    if (files.length > 100) {
      setOperationError(
        "Choose up to 100 images at a time. No files were added.",
      );
      return;
    }
    if (files.some((file) => file.size > 20 * 1024 * 1024)) {
      setOperationError(
        "Each image must be 20 MB or smaller. Choose smaller files and try again.",
      );
      return;
    }
    camera.stop();
    setActivity("analyze");
    setMediaType("image");
    setOperationError(null);
    setItems(
      files.map((file, index) => ({
        id: `${Date.now()}-${index}`,
        filename: file.name,
        file,
        previewUrl: rememberUrl(file),
        imageType,
        rotation: 0,
      })),
    );
    setSelectedId(null);
    setReviewIndex(0);
    setPreparing(true);
    setProgress(0);
  }
  function updatePrepared(update: Partial<AnalysisItem>) {
    setItems((current) =>
      current.map((item, index) =>
        index === reviewIndex ? { ...item, ...update } : item,
      ),
    );
  }
  async function analyzeImages(retry = false) {
    if (running.current || !online) return;
    running.current = true;
    setBusy(true);
    setPreparing(false);
    setOperationError(null);
    setProgress(0);
    const queue = items.map((item) => ({ ...item }));
    const pending = retry ? queue.filter((item) => item.error) : queue;
    if (retry) setProgress(queue.length - pending.length);
    for (const item of pending) {
      if (!mounted.current) break;
      try {
        const oriented = await bakeImageRotation(item.file, item.rotation);
        const response = await (item.imageType === "close_up"
          ? inferCloseUpImage(oriented)
          : inferImage(oriented));
        if (!mounted.current) break;
        if (oriented !== item.file) {
          item.previewUrl = rememberUrl(oriented);
          item.file = oriented;
        }
        item.rotation = 0;
        item.result = response.result;
        item.error = undefined;
        item.runtimeLabel =
          response.result.events[0]?.device ??
          health?.models.detector_device ??
          "Default device";
        item.detectorRuntime =
          item.imageType === "close_up"
            ? "Bypassed · whole image"
            : (health?.models.detector ?? "Unavailable");
        item.classifierRuntime =
          item.imageType === "close_up"
            ? "Close-up compatibility candidate"
            : (health?.models.classifier ?? "Unavailable");
        item.modelWarnings = [
          ...modelWarnings,
          ...response.result.warnings,
          ...(item.imageType === "close_up"
            ? ["Close-up mode uses the isolated compatibility candidate."]
            : []),
        ];
        setRecent((current) =>
          [
            { ...item },
            ...current.filter((existing) => existing.id !== item.id),
          ].slice(0, 24),
        );
      } catch (error) {
        item.error =
          error instanceof Error
            ? error.message
            : "Could not analyze this image.";
      }
      if (mounted.current) {
        setItems(queue.map((value) => ({ ...value })));
        setProgress((current) => current + 1);
      }
    }
    if (mounted.current) {
      setBusy(false);
      if (queue.length === 1) {
        setSelectedId(queue[0].id);
        if (queue[0].error) setOperationError(queue[0].error);
      }
    }
    running.current = false;
  }
  async function uploadVideo(file: File) {
    if (running.current) return;
    if (file.size > 250 * 1024 * 1024) {
      setOperationError("Video must be 250 MB or smaller.");
      return;
    }
    camera.stop();
    setActivity("analyze");
    setMediaType("video");
    setBusy(true);
    running.current = true;
    setVideoUrl(URL.createObjectURL(file));
    setVideoSummary(null);
    setVideoFailed(false);
    setOperationError(null);
    try {
      const response = await inferVideo(file, progress => { if(mounted.current) setVideoProgress(progress); });
      if (mounted.current) setVideoSummary(response);
    } catch (error) {
      if (mounted.current) {
        setVideoFailed(true);
        setOperationError(
          error instanceof Error ? error.message : "Video analysis failed.",
        );
      }
    } finally {
      running.current = false;
      if (mounted.current) setBusy(false);
    }
  }
  function openRecent(item: AnalysisItem) {
    setItems([item]);
    setSelectedId(item.id);
    setPreparing(false);
    setImageType(item.imageType);
    setMediaType("image");
    setActivity("analyze");
    setOperationError(null);
  }
  const preparation = items[reviewIndex];
  return (
    <main className={`night-app ${presenterMode ? "focus-mode" : ""}`}>
      <header className="night-header">
        <div className="brand night-brand">
          <span className="night-brand-mark">
            <Route size={23} />
          </span>
          <div>
            <h1>RoadSign Assist</h1>
            <span>Malaysian road intelligence</span>
          </div>
        </div>
        <div className="night-header-tools">
          <span className={`connection-state ${online ? "ready" : "offline"}`}>
            <i />
            {online
              ? "System ready"
              : healthError
                ? "Backend offline"
                : "Connecting"}
          </span>
          <span className="runtime-badge">
            {health?.models.runtime_badge ?? "CHECKING"}
          </span>
          <button
            className="icon-button"
            onClick={() => setPresenterMode((value) => !value)}
            aria-label={
              presenterMode ? "Exit presenter mode" : "Presenter mode"
            }
            title="Focus view"
          >
            {presenterMode ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
          </button>
        </div>
      </header>
      <MovableDock>
        <button
          className="lens-add"
          disabled={busy || !online}
          aria-label="Add media"
          onClick={() =>
            mediaType === "video"
              ? videoInput.current?.click()
              : fileInput.current?.click()
          }
        >
          <Upload size={20} />
          <span>Add</span>
        </button>
        <button
          aria-current={
            activity === "analyze" && mediaType === "image" ? "page" : undefined
          }
          disabled={busy}
          onClick={() =>
            transition(() => {
              camera.stop();
              setCameraResult(null);
              setOperationError(null);
              setMediaType("image");
              setActivity("analyze");
            })
          }
        >
          <ImagePlus size={20} />
          <span>Images</span>
        </button>
        <button
          aria-current={
            activity === "analyze" && mediaType === "video" ? "page" : undefined
          }
          disabled={busy}
          onClick={() =>
            transition(() => {
              camera.stop();
              setCameraResult(null);
              setOperationError(null);
              setMediaType("video");
              setActivity("analyze");
            })
          }
        >
          <Film size={20} />
          <span>Video</span>
        </button>
        <button
          aria-current={activity === "live" ? "page" : undefined}
          disabled={busy}
          onClick={() => navigate("live")}
        >
          <Radio size={20} />
          <span>Live</span>
        </button>
        <button
          aria-current={activity === "recent" ? "page" : undefined}
          disabled={busy}
          onClick={() => navigate("recent")}
        >
          <History size={20} />
          <span>Recent</span>
        </button>
      </MovableDock>
      <div className="night-body">
        <div className="workspace-heading">
          <div className="language-row">
            <label className="language-control">
              <Languages size={16} />
              <select
                aria-label="Warning language"
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
            <button
              className="icon-button"
              onClick={() => setMuted((value) => !value)}
              aria-label={muted ? "Enable warnings" : "Mute warnings"}
              title={muted ? "Enable warnings" : "Mute warnings"}
            >
              {muted ? <VolumeX size={18} /> : <Volume2 size={18} />}
            </button>
          </div>
        </div>
        {(operationError ||
          healthError ||
          audio.error ||
          (activity === "live" && liveSource === "camera" && camera.error)) && (
          <div role="alert" className="error-banner">
            {operationError || healthError || audio.error || camera.error}
            {audio.blocked && (
              <button onClick={audio.enable}>Enable audio</button>
            )}
            {healthError && (
              <button
                className="quiet-button"
                onClick={() => void refreshHealth()}
              >
                Reconnect
              </button>
            )}
          </div>
        )}
        <input
          ref={fileInput}
          className="sr-only"
          type="file"
          accept={imageAccept}
          multiple
          disabled={busy || !online}
          onChange={(event) => {
            uploadImages(Array.from(event.target.files ?? []));
            event.target.value = "";
          }}
        />
        <input
          ref={videoInput}
          className="sr-only"
          type="file"
          accept="video/mp4,video/webm,video/quicktime,video/x-msvideo"
          disabled={busy || !online}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void uploadVideo(file);
            event.target.value = "";
          }}
        />
        <div className="activity-transition" key={activity}>
          {activity === "analyze" && (
            <section
              className="night-workspace"
              aria-label="Analysis workspace"
            >
              <div className="context-toolbar">
                {mediaType === "image" && !preparing && !selected?.result && (
                  <div
                    className="pill-tabs image-type-tabs"
                    aria-label="Image analysis mode"
                  >
                    <button
                      disabled={busy}
                      aria-pressed={imageType === "road_scene"}
                      className={imageType === "road_scene" ? "active" : ""}
                      onClick={() => setImageType("road_scene")}
                    >
                      <Route size={16} />
                      Road scene
                    </button>
                    <button
                      disabled={busy}
                      aria-pressed={imageType === "close_up"}
                      className={imageType === "close_up" ? "active" : ""}
                      onClick={() => setImageType("close_up")}
                    >
                      <ScanLine size={16} />
                      Close-up sign
                    </button>
                  </div>
                )}
                {selected?.result && mediaType === "image" && (
                  <span className="type-badge">
                    {typeLabel(selected.imageType)}
                  </span>
                )}
                <button
                  className="quiet-button upload-command"
                  disabled={busy || !online}
                  aria-label={
                    mediaType === "image" ? "Choose image" : "Choose video"
                  }
                  onClick={() =>
                    mediaType === "image"
                      ? fileInput.current?.click()
                      : videoInput.current?.click()
                  }
                >
                  <Upload size={16} />
                  {mediaType === "image" ? "Add images" : "Add video"}
                </button>
              </div>
              <div className="media-transition" key={mediaType}>
                {mediaType === "video" ? (
                  <VideoResults
                    progress={videoProgress}
                    videoUrl={videoUrl}
                    summary={videoSummary}
                    busy={busy}
                    failed={videoFailed}
                    muted={muted}
                    language={language}
                    onChoose={() => videoInput.current?.click()}
                    disabled={!online}
                  />
                ) : preparing && preparation ? (
                  <section
                    className="prepare-workspace"
                    aria-label="Review image orientation"
                  >
                    <div className="prepare-heading">
                      <div>
                        <span className="eyebrow">
                          PREPARE · {reviewIndex + 1} OF {items.length}
                        </span>
                        <h3>
                          {preparation.imageType === "close_up"
                            ? "Make sure the sign is upright"
                            : "Make sure the road scene is upright"}
                        </h3>
                        <p>
                          Check each image’s type and orientation before
                          analysis.
                        </p>
                      </div>
                      <button
                        className="icon-button"
                        aria-label="Discard selection"
                        onClick={() => {
                          setPreparing(false);
                          setItems([]);
                        }}
                      >
                        <X size={18} />
                      </button>
                    </div>
                    <div className="prepare-layout">
                      <div className="orientation-review-media">
                        <img
                          src={preparation.previewUrl}
                          alt={
                            preparation.imageType === "close_up"
                              ? "Selected close-up sign awaiting orientation confirmation"
                              : "Selected road scene awaiting orientation confirmation"
                          }
                          style={{
                            transform: `rotate(${preparation.rotation}deg)`,
                            maxWidth:
                              preparation.rotation % 180 ? "55%" : "90%",
                            maxHeight: "80%",
                          }}
                        />
                      </div>
                      <aside className="prepare-inspector">
                        <span className="eyebrow">IMAGE TYPE</span>
                        <div className="type-cards">
                          {(["road_scene", "close_up"] as const).map((type) => (
                            <button
                              key={type}
                              className={
                                preparation.imageType === type ? "selected" : ""
                              }
                              aria-pressed={preparation.imageType === type}
                              onClick={() =>
                                updatePrepared({ imageType: type })
                              }
                            >
                              {type === "road_scene" ? (
                                <Route size={22} />
                              ) : (
                                <ScanLine size={22} />
                              )}
                              <span>
                                <strong>{typeLabel(type)}</strong>
                                <small>
                                  {type === "road_scene"
                                    ? "Find signs in a full road photo."
                                    : "One upright sign fills the image."}
                                </small>
                              </span>
                            </button>
                          ))}
                        </div>
                        <span className="eyebrow">ORIENTATION</span>
                        <div className="orientation-review-controls">
                          <button
                            aria-label="Rotate left"
                            onClick={() =>
                              updatePrepared({
                                rotation: rotateQuarterTurn(
                                  preparation.rotation,
                                  -90,
                                ),
                              })
                            }
                          >
                            <RotateCcw size={18} />
                          </button>
                          <button
                            onClick={() => updatePrepared({ rotation: 0 })}
                            disabled={preparation.rotation === 0}
                          >
                            Reset
                          </button>
                          <button
                            aria-label="Rotate right"
                            onClick={() =>
                              updatePrepared({
                                rotation: rotateQuarterTurn(
                                  preparation.rotation,
                                  90,
                                ),
                              })
                            }
                          >
                            <RotateCw size={18} />
                          </button>
                        </div>
                        <p className="subtle-note">
                          Direction signs change meaning when sideways. Check
                          the whole image is upright.
                        </p>
                        <button
                          className="analysis-primary-action"
                          onClick={() => void analyzeImages()}
                          disabled={!online}
                        >
                          <ScanLine size={17} />
                          {items.length === 1
                            ? "Analyze upright image"
                            : `Analyze ${items.length} images`}
                        </button>
                      </aside>
                    </div>
                    <div className="preparation-strip">
                      {items.map((item, index) => (
                        <button
                          className={reviewIndex === index ? "active" : ""}
                          key={item.id}
                          onClick={() => setReviewIndex(index)}
                          aria-label={`Prepare ${item.filename}`}
                        >
                          <img src={item.previewUrl} alt="" />
                          <span>
                            {item.filename}
                            <small>
                              {typeLabel(item.imageType)} · {item.rotation}°
                            </small>
                          </span>
                        </button>
                      ))}
                    </div>
                  </section>
                ) : busy ? (
                  <div className="processing-state" role="status">
                    <AnalysisLoader />
                    <h3>Reading your images</h3>
                    <p>
                      {progress} of {items.length} processed.
                    </p>
                    <progress value={progress} max={items.length || 1} />
                  </div>
                ) : selected?.result ? (
                  <ImageAnalysisWorkspace
                    imageUrl={selected.previewUrl}
                    result={selected.result}
                    busy={false}
                    language={language}
                    runtimeLabel={selected.runtimeLabel ?? "Unknown"}
                    detectorRuntime={selected.detectorRuntime ?? "Unknown"}
                    classifierRuntime={selected.classifierRuntime ?? "Unknown"}
                    modelWarnings={selected.modelWarnings ?? []}
                    contextLabel={selected.filename}
                    closeUpMode={selected.imageType === "close_up"}
                    onChooseImage={() => fileInput.current?.click()}
                    onBackToBatch={
                      items.length > 1 ? () => setSelectedId(null) : undefined
                    }
                  />
                ) : items.length ? (
                  <>
                    <BatchResults
                      items={items}
                      busy={false}
                      language={language}
                      onOpenItem={(item) =>
                        setSelectedId((item as AnalysisItem).id)
                      }
                    />
                    {items.some((item) => item.error) && (
                      <button
                        className="quiet-button retry-batch"
                        disabled={!online}
                        onClick={() => void analyzeImages(true)}
                      >
                        <RotateCcw size={16} />
                        Retry failed images
                      </button>
                    )}
                  </>
                ) : (
                  <div className="night-empty immersive-home">
                    <div className="home-road-world" aria-hidden="true">
                      <svg viewBox="0 0 800 900" fill="none">
                        <defs>
                          <linearGradient
                            id="road-light"
                            x1="400"
                            y1="0"
                            x2="400"
                            y2="900"
                            gradientUnits="userSpaceOnUse"
                          >
                            <stop stopColor="#d5ed9c" stopOpacity=".04" />
                            <stop
                              offset="1"
                              stopColor="#d5ed9c"
                              stopOpacity=".5"
                            />
                          </linearGradient>
                        </defs>
                        <path
                          d="M390 0C390 350 20 430 90 900M450 0C450 350 730 430 740 900"
                          stroke="url(#road-light)"
                          strokeWidth="2"
                        />
                        <path
                          d="M420 0C420 350 375 470 410 900"
                          stroke="url(#road-light)"
                          strokeWidth="3"
                          strokeDasharray="36 38"
                        />
                        <circle
                          cx="440"
                          cy="320"
                          r="190"
                          stroke="#d5ed9c"
                          strokeOpacity=".09"
                        />
                        <circle
                          cx="440"
                          cy="320"
                          r="270"
                          stroke="#d5ed9c"
                          strokeOpacity=".05"
                        />
                      </svg>
                      <div className="home-sign-object">
                        <span>30</span>
                        <small>km/h</small>
                      </div>
                      <span className="home-object-caption">
                        A clearer view starts here.
                      </span>
                    </div>
                    <span className="eyebrow">
                      {imageType === "road_scene"
                        ? "A WIDER PERSPECTIVE"
                        : "A CLOSER LOOK"}
                    </span>
                    <h3>
                      {imageType === "road_scene"
                        ? "Every sign tells a story."
                        : "One sign. A clearer meaning."}
                    </h3>
                    <p>
                      {imageType === "road_scene"
                        ? "Drop in a road photo. Discover the signs."
                        : "One upright sign. A focused scan."}
                    </p>
                    <button
                      className="analysis-primary-action"
                      disabled={!online}
                      onClick={() => fileInput.current?.click()}
                    >
                      <Upload size={18} />
                      Choose images
                      <ArrowUpRight size={16} />
                    </button>
                    <span className="file-help">
                      JPG, PNG, WEBP, BMP · Up to 100 images · 20 MB each
                    </span>
                    <div className="workflow-footnotes">
                      <span>
                        <ImagePlus size={16} />
                        Add one or many
                      </span>
                      <ChevronRight size={14} />
                      <span>
                        <RotateCw size={16} />
                        Check orientation
                      </span>
                      <ChevronRight size={14} />
                      <span>
                        <ScanLine size={16} />
                        Explore findings
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </section>
          )}
          {activity === "live" && (
            <section className="night-workspace" aria-label="Live workspace">
              <div className="context-toolbar">
                <div className="pill-tabs">
                  <button
                    className={liveSource === "camera" ? "active" : ""}
                    disabled={publicHost}
                    onClick={() => {
                      setLiveSource("camera");
                      setShowPairing(false);
                    }}
                  >
                    <Camera size={17} />
                    This device
                  </button>
                  <button
                    className={liveSource === "phone" ? "active" : ""}
                    onClick={() => {
                      camera.stop();
                      setCameraResult(null);
                      setLiveSource("phone");
                    }}
                  >
                    <Smartphone size={17} />
                    Connected phones
                  </button>
                </div>
                {liveSource === "phone" ? (
                  <button
                    className="quiet-button upload-command"
                    aria-expanded={showPairing}
                    onClick={() => setShowPairing((value) => !value)}
                  >
                    <Smartphone size={17} />
                    {showPairing ? "Close pairing" : "Add phone"}
                  </button>
                ) : (
                  <button
                    className="quiet-button upload-command"
                    disabled={
                      !online &&
                      camera.status !== "live" &&
                      camera.status !== "connecting"
                    }
                    onClick={() => {
                      if (
                        camera.status === "live" ||
                        camera.status === "connecting"
                      ) {
                        camera.stop();
                        setCameraResult(null);
                      } else void camera.start();
                    }}
                  >
                    {camera.status === "live" ||
                    camera.status === "connecting" ? (
                      <>
                        <CameraOff size={17} />
                        Stop camera
                      </>
                    ) : (
                      <>
                        <Camera size={17} />
                        Start camera
                      </>
                    )}
                  </button>
                )}
              </div>
              {liveSource === "phone" ? (
                <>
                  {showPairing && <PhoneConnectPanel busy={false} />}
                  <LiveCameraWallApp
                    embedded
                    language={language}
                    muted={muted}
                    onMutedChange={setMuted}
                  />
                </>
              ) : (
                <>
                  <div className="camera-workspace">
                    <div className="camera-stage">
                      <VideoSurface
                        mode="camera"
                        videoRef={camera.videoRef}
                        imageUrl={null}
                        result={cameraResult}
                        language={language}
                      />
                      {camera.status !== "live" && (
                        <div className="camera-idle">
                          <Camera size={34} />
                          <h3>
                            {camera.status === "connecting"
                              ? "Connecting your camera…"
                              : "Your next view starts here."}
                          </h3>
                          <p>
                            Allow camera access, then point it toward a road
                            sign.
                          </p>
                        </div>
                      )}
                    </div>
                    <aside className="live-findings">
                      <EncounterPanel
                        state={cameraFindings}
                        raw={cameraResult}
                        language={language}
                      />
                      {audio.blocked && (
                        <button onClick={audio.enable}>Enable audio</button>
                      )}
                      <EventTimeline
                        events={cameraFindings.history.map((e) => e.event)}
                        language={language}
                      />
                    </aside>
                  </div>
                  <details className="stream-details">
                    <summary>Camera details · {camera.status}</summary>
                    <p>
                      Inference {camera.stats.inferenceFps} FPS ·{" "}
                      {camera.stats.latencyMs ?? "—"} ms ·{" "}
                      {camera.stats.framesDropped} dropped frames
                    </p>
                    <p>Changing source stops this device’s camera.</p>
                  </details>
                </>
              )}
            </section>
          )}
          {activity === "recent" && (
            <section
              className="night-workspace recent-workspace"
              aria-label="Recent analyses"
            >
              <div className="context-toolbar">
                <span>
                  <History size={17} />
                  This session · {recent.length} images
                </span>
                {recent.length > 0 && (
                  <button
                    className="quiet-button upload-command"
                    onClick={() => setRecent([])}
                  >
                    Clear recent
                  </button>
                )}
              </div>
              {recent.length ? (
                <div className="recent-grid">
                  {recent.map((item) => (
                    <button
                      key={item.id}
                      className="recent-card"
                      onClick={() => openRecent(item)}
                    >
                      <img src={item.previewUrl} alt="" />
                      <span className="type-badge">
                        {typeLabel(item.imageType)}
                      </span>
                      <strong>
                        {item.result?.events[0]
                          ? semanticSignName(item.result.events[0], language)
                          : "No sign detected"}
                      </strong>
                      <small>{item.filename}</small>
                      <span className="recent-open">
                        View findings
                        <ArrowUpRight size={16} />
                      </span>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="night-empty compact">
                  <History size={38} />
                  <h3>A fresh perspective.</h3>
                  <p>
                    Your analyzed images will appear here during this session.
                  </p>
                  <button
                    className="quiet-button"
                    onClick={() => navigate("analyze")}
                  >
                    Go to Analyze
                    <ChevronRight size={16} />
                  </button>
                </div>
              )}
            </section>
          )}
        </div>
        <SystemStatus health={health} refresh={() => void refreshHealth()} />
      </div>
    </main>
  );
}

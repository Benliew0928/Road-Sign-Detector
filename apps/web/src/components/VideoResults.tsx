import type { VideoProgress } from "../api";
import { AnalysisLoader } from "./AnalysisLoader";
import { Film, ScanLine } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { semanticSignName } from "../advisoryDisplay";
import type {
  DisplayLanguage,
  FrameResult,
  SignEvent,
  VideoInferenceResponse,
} from "../types";
import { LensCanvas } from "./LensCanvas";
import { LensDetails } from "./LensDetails";
import { automaticVideoSign, videoSigns } from "../lens";
import { buildEncounterTimeline, encounterAt } from "../encounters";
import { useAdvisoryAudio } from "../hooks/useAdvisoryAudio";
interface VideoResultsProps {
  progress?: VideoProgress | null;
  failed?: boolean;
  language?: DisplayLanguage;
  onChoose?: () => void;
  disabled?: boolean;
  muted?: boolean;
  videoUrl: string | null;
  summary: VideoInferenceResponse | null;
  busy: boolean;
}
function resultForFrame(
  summary: VideoInferenceResponse | null,
  sourceFrame: number,
): FrameResult | null {
  if (!summary) return null;
  const frameResults = summary.frame_results ?? [];
  for (let index = frameResults.length - 1; index >= 0; index -= 1) {
    if (frameResults[index].source_frame <= sourceFrame) {
      return frameResults[index].result;
    }
  }
  return frameResults.length ? null : summary.representative_result;
}
export function VideoResults({
  progress,
  videoUrl,
  summary,
  busy,
  failed = false,
  language = "en",
  onChoose,
  disabled,
  muted = false,
}: VideoResultsProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [selection, setSelection] = useState<{
    source: string;
    frame: number;
    event: SignEvent;
  } | null>(null);
  const [playback, setPlayback] = useState({
    videoUrl: null as string | null,
    frame: 0,
  });
  const fps = summary?.fps && summary.fps > 0 ? summary.fps : 30;
  const playbackFrame = playback.videoUrl === videoUrl ? playback.frame : 0;
  const activeResult = useMemo(
    () => resultForFrame(summary, playbackFrame),
    [playbackFrame, summary],
  );
  const source = videoUrl ?? "video";
  const [playing, setPlaying] = useState(false);
  const [seeking, setSeeking] = useState(false);
  const [resetToken, setResetToken] = useState(0);
  const [skipBefore, setSkipBefore] = useState(-Infinity);
  const replaying = useRef(false);
  const indexed = useMemo(
    () => buildEncounterTimeline(summary, source),
    [summary, source],
  );
  const findings = useMemo(
    () => encounterAt(indexed.timeline, (playbackFrame / fps) * 1000, source),
    [indexed, playbackFrame, fps, source],
  );
  const audio = useAdvisoryAudio({
    encounters: findings,
    source,
    language,
    muted,
    enabled: playing && !seeking,
    resetToken,
    skipBefore,
  });
  const [previousSource, setPreviousSource] = useState(source);
  if (previousSource !== source) {
    setPreviousSource(source);
    setPlaying(false);
    setSeeking(false);
    setSkipBefore(-Infinity);
  }
  useEffect(() => {
    let animationFrame = 0;
    const updatePlaybackFrame = () => {
      const video = videoRef.current;
      if (video) {
        const nextFrame = Math.max(0, Math.floor(video.currentTime * fps));
        setPlayback((current) =>
          current.videoUrl === videoUrl && current.frame === nextFrame
            ? current
            : { videoUrl, frame: nextFrame },
        );
      }
      animationFrame = window.requestAnimationFrame(updatePlaybackFrame);
    };
    animationFrame = window.requestAnimationFrame(updatePlaybackFrame);
    return () => window.cancelAnimationFrame(animationFrame);
  }, [fps, videoUrl]);
  const focus =
    !playing &&
    !seeking &&
    selection?.source === source &&
    selection.frame === playbackFrame
      ? selection.event
      : null;
  const automatic =
    playing && !seeking ? automaticVideoSign(activeResult) : null;
  const visibleSigns = playing
    ? videoSigns(activeResult)
    : (activeResult?.events ?? []).filter(
        (event) => event.semantic_sign_id !== "unknown_sign",
      );
  useEffect(() => {
    const reset = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelection(null);
    };
    window.addEventListener("keydown", reset);
    return () => window.removeEventListener("keydown", reset);
  }, []);
  if (!videoUrl) {
    return (
      <section className="night-empty" aria-label="Video analysis">
        <Film size={34} aria-hidden="true" />
        <span className="eyebrow">THE ROAD IN MOTION</span>
        <h3>A new perspective, frame by frame.</h3>
        <p>
          Add a road video to explore signs and review the moments that matter.
        </p>
        <button
          className="analysis-primary-action"
          onClick={onChoose}
          disabled={disabled}
        >
          <Film size={17} />
          Choose video
        </button>
        <span className="file-help">MP4, WEBM, MOV, AVI · Up to 250 MB</span>
      </section>
    );
  }
  return (
    <section className="video-analysis" aria-label="Video analysis">
      {audio.error && (
        <div className="error-banner" role="alert">
          {audio.error}
        </div>
      )}
      <LensCanvas
        result={activeResult}
        focus={focus}
        events={visibleSigns}
        language={language}
        label="Video canvas"
        onSelect={
          playing || seeking
            ? undefined
            : (event) => setSelection({ source, frame: playbackFrame, event })
        }
        onOverview={() => setSelection(null)}
      >
        <video
          key={videoUrl}
          ref={videoRef}
          src={videoUrl}
          controls={!focus}
          onPlay={() => {
            setSelection(null);
            setPlaying(true);
          }}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onSeeking={() => {
            setSelection(null);
            setSeeking(true);
            if (!replaying.current)
              setSkipBefore((value) =>
                Math.max(
                  value,
                  findings.at,
                  (videoRef.current?.currentTime ?? 0) * 1000,
                ),
              );
          }}
          onSeeked={() => {
            if (!replaying.current)
              setSkipBefore((value) =>
                Math.max(value, (videoRef.current?.currentTime ?? 0) * 1000),
              );
            replaying.current = false;
            setSeeking(false);
          }}
          preload="metadata"
          className="video-player lens-media"
        />
      </LensCanvas>
      {focus && (
        <button
          className="lens-overview lens-control"
          onClick={() => setSelection(null)}
        >
          Overview
        </button>
      )}
      {(focus || automatic) && (
        <LensDetails
          event={(focus || automatic)!}
          language={language}
          automatic={Boolean(automatic)}
          onClose={focus ? () => setSelection(null) : undefined}
        />
      )}
      <div className="lens-video-actions">
        {focus && (
          <button
            onClick={() => {
              setSelection(null);
              void videoRef.current?.play().catch(() => setPlaying(false));
            }}
          >
            Resume video
          </button>
        )}
        {audio.blocked && <button onClick={audio.enable}>Enable audio</button>}
        <button
          className="quiet-button"
          onClick={() => {
            const video = videoRef.current;
            if (!video) return;
            replaying.current = true;
            setSkipBefore(-Infinity);
            setResetToken((value) => value + 1);
            video.currentTime = 0;
            setPlayback({ videoUrl, frame: 0 });
            void video.play().catch(() => setPlaying(false));
          }}
        >
          Replay guidance
        </button>
        <p className="subtle-note">
          Findings follow the current playback position.
        </p>
      </div>
      <div className="video-summary">
        {busy && <AnalysisLoader />}
        <span role="status">
          {busy
            ? "Analyzing video — this may take a moment"
            : failed
              ? "Analysis failed"
              : summary
                ? "Analysis complete"
                : "Waiting for analysis"}
        </span>
        {busy && <div className="video-live-progress" role="status">
          {progress?.stage === "analyzing" ? <>
            <div><strong>{progress.total ? `${Math.min(99,Math.floor(progress.processed / progress.total * 100))}%` : `${progress.processed} frames`}</strong><span>{progress.total ? `${progress.processed.toLocaleString()} / ${progress.total.toLocaleString()} frames` : "Counting frames"}</span></div>
            <progress aria-label="Video analysis progress" value={progress.total ? Math.min(progress.processed,progress.total) : undefined} max={progress.total || 1}/>
            <small>{progress.eta_seconds == null ? "Estimating time remaining…" : progress.eta_seconds < 1 ? "Finalizing…" : `About ${progress.eta_seconds < 60 ? `${Math.ceil(progress.eta_seconds)} sec` : `${Math.ceil(progress.eta_seconds / 60)} min`} remaining`}</small>
          </> : <small>Uploading & preparing video…</small>}
        </div>}
        <dl>
          <div>
            <dt>Frames read</dt>
            <dd>{summary?.frames_read ?? "—"}</dd>
          </div>
          <div>
            <dt>Frames sampled</dt>
            <dd>{summary?.sampled_frames ?? "—"}</dd>
          </div>
          <div>
            <dt>Confirmed encounters</dt>
            <dd>{summary ? indexed.moments.length : "—"}</dd>
          </div>
        </dl>
        <p>
          <ScanLine size={14} />
          Detections update as the uploaded video plays.
        </p>
      </div>
      {summary?.frame_results && (
        <details className="video-moments">
          <summary>Explore sign moments</summary>
          <div>
            {indexed.moments.map((encounter) => (
              <button
                className="quiet-button"
                key={encounter.id}
                onClick={() => {
                  if (videoRef.current)
                    videoRef.current.currentTime = encounter.confirmedAt / 1000;
                  setPlayback({
                    videoUrl,
                    frame: Math.floor((encounter.confirmedAt / 1000) * fps),
                  });
                }}
              >
                {(encounter.confirmedAt / 1000).toFixed(1)}s ·{" "}
                {semanticSignName(encounter.event, language)}
              </button>
            ))}
            <p>Raw frame detections: {summary.events}</p>
          </div>
        </details>
      )}
    </section>
  );
}

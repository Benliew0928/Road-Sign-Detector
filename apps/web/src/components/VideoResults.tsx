import { Film, ScanLine } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";

import { advisoryHeadline } from "../advisoryDisplay";
import type { FrameResult, VideoInferenceResponse } from "../types";

interface VideoResultsProps {
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
  return frameResults[0]?.result ?? summary.representative_result;
}

export function VideoResults({ videoUrl, summary, busy }: VideoResultsProps) {
  const playerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [mediaBox, setMediaBox] = useState<CSSProperties>({});
  const [playback, setPlayback] = useState({ videoUrl: null as string | null, frame: 0 });
  const fps = summary?.fps && summary.fps > 0 ? summary.fps : 30;
  const playbackFrame = playback.videoUrl === videoUrl ? playback.frame : 0;
  const activeResult = useMemo(
    () => resultForFrame(summary, playbackFrame),
    [playbackFrame, summary],
  );

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

  useEffect(() => {
    const player = playerRef.current;
    if (!player || !activeResult) {
      setMediaBox({});
      return;
    }

    const updateMediaBox = () => {
      const width = player.clientWidth;
      const height = player.clientHeight;
      if (!width || !height) return;
      const mediaRatio = activeResult.width / activeResult.height;
      const playerRatio = width / height;
      if (playerRatio > mediaRatio) {
        const renderedWidth = height * mediaRatio;
        setMediaBox({
          left: (width - renderedWidth) / 2,
          top: 0,
          width: renderedWidth,
          height,
        });
      } else {
        const renderedHeight = width / mediaRatio;
        setMediaBox({
          left: 0,
          top: (height - renderedHeight) / 2,
          width,
          height: renderedHeight,
        });
      }
    };

    updateMediaBox();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(updateMediaBox);
    observer.observe(player);
    return () => observer.disconnect();
  }, [activeResult]);

  if (!videoUrl) {
    return (
      <section className="batch-empty" aria-label="Video analysis">
        <Film size={34} aria-hidden="true" />
        <strong>Select a road video</strong>
        <span>Every decoded frame will be scanned at the video's original resolution.</span>
      </section>
    );
  }

  return (
    <section className="video-analysis" aria-label="Video analysis">
      <div ref={playerRef} className="video-player-wrap">
        <div className="video-media-space" style={activeResult ? mediaBox : { inset: 0 }}>
          <video
            key={videoUrl}
            ref={videoRef}
            src={videoUrl}
            controls
            preload="metadata"
            className="video-player"
          />
          {activeResult ? (
            <div
              className="video-overlay-layer"
              aria-label={`${activeResult.events.length} detected signs in the current video frame`}
            >
              {activeResult.events.map((event) => (
                <div
                  className={`detection-box severity-${event.severity}`}
                  key={`${activeResult.frame_id}-${event.track_id}`}
                  style={{
                    left: `${(event.bbox.x1 / activeResult.width) * 100}%`,
                    top: `${(event.bbox.y1 / activeResult.height) * 100}%`,
                    width: `${((event.bbox.x2 - event.bbox.x1) / activeResult.width) * 100}%`,
                    height: `${((event.bbox.y2 - event.bbox.y1) / activeResult.height) * 100}%`,
                  }}
                >
                  <span className="detection-label">
                    #{event.track_id} {advisoryHeadline(event, "en")} {Math.round(event.confidence * 100)}%
                  </span>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      </div>
      <div className="video-summary">
        <span>{busy ? "Analyzing video" : "Analysis complete"}</span>
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
            <dt>Sign events</dt>
            <dd>{summary?.events ?? "—"}</dd>
          </div>
        </dl>
        <p>
          <ScanLine size={14} />
          Detections update as the uploaded video plays.
        </p>
      </div>
    </section>
  );
}

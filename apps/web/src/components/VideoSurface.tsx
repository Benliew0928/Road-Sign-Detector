import { ImageIcon, ScanLine } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { FrameResult, SourceMode } from "../types";

interface VideoSurfaceProps {
  mode: SourceMode;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  imageUrl: string | null;
  result: FrameResult | null;
}

export function VideoSurface({ mode, videoRef, imageUrl, result }: VideoSurfaceProps) {
  const hasMedia = mode === "camera" || (mode === "image" && imageUrl);
  const surfaceRef = useRef<HTMLElement>(null);
  const [mediaBox, setMediaBox] = useState<React.CSSProperties>({});

  useEffect(() => {
    const surface = surfaceRef.current;
    if (!surface || !result) {
      setMediaBox({});
      return;
    }
    const update = () => {
      const width = surface.clientWidth;
      const height = surface.clientHeight;
      const mediaRatio = result.width / result.height;
      const surfaceRatio = width / height;
      if (surfaceRatio > mediaRatio) {
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
    update();
    const observer = new ResizeObserver(update);
    observer.observe(surface);
    return () => observer.disconnect();
  }, [result]);

  return (
    <section ref={surfaceRef} className="video-surface" aria-label="Live road sign view">
      {!hasMedia ? (
        <div className="empty-media">
          <ImageIcon size={34} aria-hidden="true" />
          <span>Select an image</span>
        </div>
      ) : (
        <div className="media-coordinate-space" style={result ? mediaBox : { inset: 0 }}>
          {mode === "camera" ? (
            <video
              ref={videoRef}
              muted
              playsInline
              className={`media-layer ${result ? "fitted" : ""}`}
            />
          ) : (
            <img
              src={imageUrl ?? ""}
              alt={result ? "Analyzed road scene" : "Selected road scene"}
              className={`media-layer ${result ? "fitted" : ""}`}
            />
          )}
          {result ? (
            <>
              <svg
                className="mask-layer"
                viewBox={`0 0 ${result.width} ${result.height}`}
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                {result.events.map((event) =>
                  event.mask?.points.length ? (
                    <polygon
                      key={`mask-${event.frame_id}-${event.track_id}`}
                      className={`segmentation-mask severity-${event.severity}`}
                      points={event.mask.points.map(([x, y]) => `${x},${y}`).join(" ")}
                    />
                  ) : null,
                )}
              </svg>
              <div
                className="overlay-layer"
                aria-label={`${result.events.length} detected signs`}
              >
                {result.events.map((event) => (
                  <div
                    className={`detection-box severity-${event.severity}`}
                    key={`${event.frame_id}-${event.track_id}`}
                    style={{
                      left: `${(event.bbox.x1 / result.width) * 100}%`,
                      top: `${(event.bbox.y1 / result.height) * 100}%`,
                      width: `${((event.bbox.x2 - event.bbox.x1) / result.width) * 100}%`,
                      height: `${((event.bbox.y2 - event.bbox.y1) / result.height) * 100}%`,
                    }}
                  >
                    <span className="detection-label">
                      #{event.track_id} {event.meaning.en}{" "}
                      {Math.round(event.confidence * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            </>
          ) : null}
        </div>
      )}

      <div className="viewfinder" aria-hidden="true">
        <ScanLine size={24} />
      </div>
    </section>
  );
}

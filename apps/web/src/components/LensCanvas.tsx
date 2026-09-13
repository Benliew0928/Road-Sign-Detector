import { useEffect, useRef, useState, type ReactNode } from "react";
import { focusTransform } from "../lens";
import { semanticSignName } from "../advisoryDisplay";
import type { DisplayLanguage, FrameResult, SignEvent } from "../types";
export function LensCanvas({
  result,
  focus,
  events,
  language,
  onSelect,
  children,
  label,
  showBoxes = true,
  onOverview,
}: {
  result: FrameResult | null;
  focus: SignEvent | null;
  events: SignEvent[];
  language: DisplayLanguage;
  onSelect?: (event: SignEvent) => void;
  children: ReactNode;
  label: string;
  showBoxes?: boolean;
  onOverview?: () => void;
}) {
  const [fitAll, setFitAll] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 1, height: 1 });
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const update = () =>
      setSize({ width: element.clientWidth, height: element.clientHeight });
    update();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const geometry = focusTransform(
    size,
    result ?? size,
    focus?.bbox ?? null,
    fitAll,
  );
  const { zoom, ...planeStyle } = geometry;
  const style = result ? planeStyle : { inset: 0 };
  const narrow = size.width < 720;
  const outside =
    !focus && result && showBoxes
      ? events.filter((event) => {
          const x1 =
            geometry.left + (event.bbox.x1 / result.width) * geometry.width;
          const x2 =
            geometry.left + (event.bbox.x2 / result.width) * geometry.width;
          const y1 =
            geometry.top + (event.bbox.y1 / result.height) * geometry.height;
          const y2 =
            geometry.top + (event.bbox.y2 / result.height) * geometry.height;
          return x1 < 0 || x2 > size.width || y1 < 0 || y2 > size.height;
        })
      : [];
  const aperture =
    focus && result
      ? `radial-gradient(ellipse ${Math.max(40, ((focus.bbox.x2 - focus.bbox.x1) / result.width) * geometry.width * zoom * 0.9 + 28)}px ${Math.max(40, ((focus.bbox.y2 - focus.bbox.y1) / result.height) * geometry.height * zoom * 0.9 + 28)}px at ${size.width * (narrow ? 0.5 : 0.4)}px ${size.height * (narrow ? 0.32 : 0.46)}px, transparent 82%, black 115%)`
      : undefined;
  return (
    <div
      ref={ref}
      className={`lens-canvas ${focus ? "is-focused" : ""}`}
      aria-label={label}
      onClick={(event) => {
        if (event.target === event.currentTarget) onOverview?.();
      }}
    >
      <div
        className="lens-media-plane"
        style={style}
        data-focused={Boolean(focus)}
      >
        {children}
        {result && showBoxes && (
          <div
            className="lens-box-layer"
            aria-label={`${events.length} detected signs`}
          >
            <svg
              className="mask-layer"
              viewBox={`0 0 ${result.width} ${result.height}`}
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              {events.map((event) =>
                event.mask?.points.length ? (
                  <polygon
                    key={event.track_id}
                    className={`segmentation-mask severity-${event.severity}`}
                    points={event.mask.points
                      .map(([x, y]) => `${x},${y}`)
                      .join(" ")}
                  />
                ) : null,
              )}
            </svg>
            {events.map((event) => {
              const position = {
                left: `${(event.bbox.x1 / result.width) * 100}%`,
                top: `${(event.bbox.y1 / result.height) * 100}%`,
                width: `${((event.bbox.x2 - event.bbox.x1) / result.width) * 100}%`,
                height: `${((event.bbox.y2 - event.bbox.y1) / result.height) * 100}%`,
              };
              const name = semanticSignName(event, language);
              return onSelect ? (
                <button
                  key={event.track_id}
                  type="button"
                  className={`lens-sign-box ${focus?.track_id === event.track_id ? "selected" : ""}`}
                  style={position}
                  aria-label={`Inspect ${name}`}
                  aria-pressed={focus?.track_id === event.track_id}
                  onClick={() => onSelect(event)}
                >
                  <span>{name}</span>
                </button>
              ) : (
                <div
                  key={event.track_id}
                  className="lens-sign-box passive"
                  style={position}
                >
                  <span>{name}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
      {outside.length > 0 && onSelect && (
        <div className="canvas-outside-signs">
          <span>Beyond this view</span>
          {outside.map((event) => (
            <button key={event.track_id} onClick={() => onSelect(event)}>
              ↗ {semanticSignName(event, language)}
            </button>
          ))}
        </div>
      )}
      {!focus && result && (
        <button
          className="canvas-fit-control"
          onClick={() => setFitAll((value) => !value)}
          aria-pressed={fitAll}
        >
          {fitAll ? "Fill screen" : "Full image"}
        </button>
      )}
      <div
        className={`lens-focus-atmosphere ${focus ? "active" : ""}`}
        aria-hidden="true"
        style={{ maskImage: aperture, WebkitMaskImage: aperture }}
      />
    </div>
  );
}

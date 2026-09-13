import type { BoundingBox, FrameResult, SignEvent } from "./types";

// Display policy only. Model acceptance and audio eligibility are unchanged.
export const VIDEO_DETAIL_CONFIDENCE = 0.8;
export function videoSigns(result: FrameResult | null): SignEvent[] {
  return (result?.events ?? []).filter(
    (event) =>
      event.semantic_sign_id !== "unknown_sign" &&
      event.confidence >= VIDEO_DETAIL_CONFIDENCE &&
      !event.evidence.some((item) => item.startsWith("tracker_hold:")),
  );
}
export function automaticVideoSign(
  result: FrameResult | null,
): SignEvent | null {
  return (
    videoSigns(result).sort(
      (a, b) =>
        Number(b.stable) - Number(a.stable) || b.confidence - a.confidence,
    )[0] ?? null
  );
}
export function focusTransform(
  view: { width: number; height: number },
  media: { width: number; height: number },
  box: BoundingBox | null,
  fitAll = false,
) {
  const fit = (fitAll ? Math.min : Math.max)(
    view.width / Math.max(1, media.width),
    view.height / Math.max(1, media.height),
  );
  const width = media.width * fit,
    height = media.height * fit;
  const left = (view.width - width) / 2,
    top = (view.height - height) / 2;
  if (!box)
    return {
      width,
      height,
      left,
      top,
      zoom: 1,
      transform: "translate(0px, 0px) scale(1)",
    };
  const narrow = view.width < 720;
  const scale = Math.max(
    1,
    Math.min(
      6,
      (view.width * (narrow ? 0.6 : 0.38)) /
        Math.max(1, (box.x2 - box.x1) * fit),
      (view.height * 0.48) / Math.max(1, (box.y2 - box.y1) * fit),
    ),
  );
  const x =
    view.width * (narrow ? 0.5 : 0.4) -
    left -
    ((box.x1 + box.x2) / 2) * fit * scale;
  const y =
    view.height * (narrow ? 0.32 : 0.46) -
    top -
    ((box.y1 + box.y2) / 2) * fit * scale;
  return {
    width,
    height,
    left,
    top,
    zoom: scale,
    transform: `translate(${x}px, ${y}px) scale(${scale})`,
  };
}

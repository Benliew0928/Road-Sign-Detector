import { X } from "lucide-react";
import {
  advisoryInstruction,
  semanticSignName,
  targetSummary,
} from "../advisoryDisplay";
import type { DisplayLanguage, FrameResult, SignEvent } from "../types";

export function LensDetails({
  event,
  language,
  imageUrl,
  result,
  onClose,
  automatic = false,
}: {
  event: SignEvent;
  language: DisplayLanguage;
  imageUrl?: string | null;
  result?: FrameResult | null;
  onClose?: () => void;
  automatic?: boolean;
}) {
  return (
    <aside
      className={`lens-details ${automatic ? "automatic" : ""}`}
      aria-label="Sign details"
      key={`${event.track_id}:${event.semantic_sign_id}`}
    >
      <div className="lens-detail-title">
        {imageUrl && result && (
          <svg
            className="lens-sign-crop"
            role="img"
            aria-label="Selected sign crop"
            viewBox={`${event.bbox.x1} ${event.bbox.y1} ${Math.max(1, event.bbox.x2 - event.bbox.x1)} ${Math.max(1, event.bbox.y2 - event.bbox.y1)}`}
          >
            <image
              href={imageUrl}
              width={result.width}
              height={result.height}
            />
          </svg>
        )}
        <div>
          <span className="lens-overline">
            {automatic ? "In this frame" : "Selected sign"}
          </span>
          <h2>{semanticSignName(event, language)}</h2>
        </div>
        {onClose && (
          <button
            className="lens-icon"
            aria-label="Close sign details"
            onClick={onClose}
          >
            <X size={17} />
          </button>
        )}
      </div>
      {event.semantic_sign_id === "unknown_sign" ? (
        <p className="lens-uncertain">Meaning uncertain</p>
      ) : (
        <>
          <strong className="lens-target">{targetSummary(event)}</strong>
          <p className="lens-advice">{advisoryInstruction(event, language)}</p>
        </>
      )}
      {event.evidence.includes("safety:directional_action_blocked") && (
        <p className="lens-uncertain">
          Verify direction. Strong actions and audio are blocked.
        </p>
      )}
      <dl>
        <div>
          <dt>Read text</dt>
          <dd>{event.ocr.text || "None"}</dd>
        </div>
        {event.semantic_sign_id !== "unknown_sign" && (
          <div>
            <dt>Model confidence</dt>
            <dd>{Math.round(event.confidence * 100)}%</dd>
          </div>
        )}
      </dl>
      <details>
        <summary>Evidence & interpretation</summary>
        <p>
          Advisory only ·{" "}
          {event.stable ? "Stable recognition" : "Single-frame finding"}
        </p>
        <p>{event.evidence.join(" · ") || "No additional evidence"}</p>
      </details>
    </aside>
  );
}

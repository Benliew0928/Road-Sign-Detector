import {
  ArrowLeft,
  Expand,
  ScanSearch,
  Timer,
  ScanLine,
  ChevronUp,
} from "lucide-react";
import { useEffect, useState } from "react";
import type { DisplayLanguage, FrameResult, SignEvent } from "../types";
import { LensCanvas } from "./LensCanvas";
import { LensDetails } from "./LensDetails";

interface ImageAnalysisWorkspaceProps {
  imageUrl: string | null;
  result: FrameResult | null;
  busy: boolean;
  language: DisplayLanguage;
  runtimeLabel: string;
  detectorRuntime: string;
  classifierRuntime: string;
  modelWarnings: string[];
  contextLabel?: string;
  onChooseImage: () => void;
  onBackToBatch?: () => void;
  closeUpMode?: boolean;
}
export function ImageAnalysisWorkspace({
  imageUrl,
  result,
  busy,
  language,
  runtimeLabel,
  modelWarnings,
  contextLabel,
  onChooseImage,
  onBackToBatch,
  closeUpMode,
}: ImageAnalysisWorkspaceProps) {
  const [selection, setSelection] = useState<{
    result: FrameResult;
    event: SignEvent;
  } | null>(null);
  const [original, setOriginal] = useState(false);
  const focus =
    selection?.result === result && !original
      ? (selection?.event ?? null)
      : null;
  useEffect(() => {
    const reset = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelection(null);
    };
    window.addEventListener("keydown", reset);
    return () => window.removeEventListener("keydown", reset);
  }, []);
  if (!result || !imageUrl)
    return (
      <div className="lens-empty">
        <ScanSearch size={32} />
        <h2>{busy ? "Analyzing image…" : "Add an image"}</h2>
        <button onClick={onChooseImage}>Choose images</button>
      </div>
    );
  return (
    <section
      className="lens-image-workspace"
      aria-label="Road-sign analysis result"
    >
      <LensCanvas
        result={result}
        focus={focus}
        events={result.events}
        language={language}
        showBoxes={!original}
        label="Image canvas"
        onSelect={(event) => setSelection({ result, event })}
        onOverview={() => setSelection(null)}
      >
        <img className="lens-media" src={imageUrl} alt="Analyzed road scene" />
      </LensCanvas>
      <div className="lens-media-tools">
        {onBackToBatch && (
          <button className="lens-control" onClick={onBackToBatch}>
            <ArrowLeft size={16} />
            Back to batch results
          </button>
        )}
        <span className="lens-filename">{contextLabel}</span>
        <span className="lens-type-label">
          {closeUpMode ? "Close-up sign" : "Road scene"}
        </span>
      </div>
      <div className="lens-view-tools">
        <div className="pill-tabs">
          <button
            className={!original ? "active" : ""}
            onClick={() => {
              setOriginal(false);
              setSelection(null);
            }}
          >
            Result
          </button>
          <button
            className={original ? "active" : ""}
            onClick={() => {
              setOriginal(true);
              setSelection(null);
            }}
          >
            Original
          </button>
        </div>
      </div>
      {focus ? (
        <>
          <button
            className="lens-overview lens-control"
            onClick={() => setSelection(null)}
          >
            <Expand size={16} />
            Overview
          </button>
          <LensDetails
            event={focus}
            language={language}
            imageUrl={imageUrl}
            result={result}
            onClose={() => setSelection(null)}
          />
        </>
      ) : (
        <span className="lens-canvas-hint">
          {result.events.length
            ? original
              ? "Original image"
              : "Select a sign to zoom in and explore"
            : "No sign detected · Try a clearer image"}
        </span>
      )}
      <details className="lens-image-evidence compact-insight">
        <summary>
          <ScanLine size={15} />
          <span>Analysis</span>
          <span className="insight-mini">
            {Math.round(result.latency_ms)} ms
          </span>
        </summary>
        <div className="insight-popover analysis-popover">
          <div className="insight-heading">
            <span className="insight-orbit">
              <ScanSearch size={22} />
            </span>
            <div>
              <small>IMAGE SNAPSHOT</small>
              <h3>{closeUpMode ? "Close-up sign" : "Road scene"}</h3>
            </div>
          </div>
          <div className="insight-metrics">
            <div>
              <strong>
                {result.events.length.toString().padStart(2, "0")}
              </strong>
              <span>Signs found</span>
            </div>
            <div>
              <strong>
                {Math.round(result.latency_ms)}
                <small>ms</small>
              </strong>
              <span>
                <Timer size={12} /> Analysis time
              </span>
            </div>
          </div>
          <div className="insight-chips">
            <span>{closeUpMode ? "Whole-image scan" : "Scene detection"}</span>
            <span>{runtimeLabel}</span>
          </div>
          <div className="insight-foot">
            <span>
              {modelWarnings.length
                ? `${modelWarnings.length} engine ${modelWarnings.length === 1 ? "notice" : "notices"}`
                : "Analysis complete"}
            </span>
            <ChevronUp size={14} />
          </div>
        </div>
      </details>
    </section>
  );
}

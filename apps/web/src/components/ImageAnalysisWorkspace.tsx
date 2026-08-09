import {
  ArrowLeft,
  BadgeCheck,
  BrainCircuit,
  ChevronDown,
  Cpu,
  Gauge,
  ImagePlus,
  ScanSearch,
  ShieldCheck,
  Shapes,
  TriangleAlert,
} from "lucide-react";
import { useMemo, useRef, useState } from "react";

import { advisoryHeadline, advisoryInstruction, targetSummary } from "../advisoryDisplay";
import type { DisplayLanguage, FrameResult, SignEvent } from "../types";
import { VideoSurface } from "./VideoSurface";

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
}

function choosePrimaryEvent(result: FrameResult | null): SignEvent | null {
  if (!result?.events.length) return null;
  return [...result.events].sort((first, second) => {
    if (first.stable !== second.stable) return first.stable ? -1 : 1;
    return second.confidence - first.confidence;
  })[0];
}

function evidenceSummary(event: SignEvent | null, prefix: string): string {
  const evidence = event?.evidence.find((item) => item.startsWith(prefix));
  if (!evidence) return "Not available";
  return evidence
    .slice(prefix.length)
    .split(":")
    .filter(Boolean)
    .join(" · ");
}

export function ImageAnalysisWorkspace({
  imageUrl,
  result,
  busy,
  language,
  runtimeLabel,
  detectorRuntime,
  classifierRuntime,
  modelWarnings,
  contextLabel,
  onChooseImage,
  onBackToBatch,
}: ImageAnalysisWorkspaceProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [sourceViewResult, setSourceViewResult] = useState<FrameResult | null>(null);
  const primaryEvent = useMemo(() => choosePrimaryEvent(result), [result]);
  const meaningUnknown = primaryEvent?.semantic_sign_id === "unknown_sign";
  const showSource = sourceViewResult === result;

  if (!result) {
    return (
      <section className="analysis-empty" aria-label="Image analysis">
        <div className="analysis-empty-icon">
          <ScanSearch size={34} aria-hidden="true" />
        </div>
        <span className="eyebrow">Road-sign analysis</span>
        <h2>{busy ? "Analyzing your image" : "Start with a road-sign image"}</h2>
        <p>
          {busy
            ? "The detection result and safety interpretation will appear here."
            : "Upload one image to detect the sign, interpret its meaning and review the evidence."}
        </p>
        {!busy ? (
          <button className="analysis-primary-action" onClick={onChooseImage}>
            <ImagePlus size={18} aria-hidden="true" />
            Upload image
          </button>
        ) : null}
      </section>
    );
  }

  const headline = primaryEvent
    ? advisoryHeadline(primaryEvent, language)
    : "No road sign detected";
  const instruction = primaryEvent
    ? advisoryInstruction(primaryEvent, language)
    : "Try an image where the sign is larger, clearer and well lit.";

  return (
    <section className="analysis-explorer" aria-label="Road-sign analysis result">
      {onBackToBatch ? (
        <button className="analysis-back-button" type="button" onClick={onBackToBatch}>
          <ArrowLeft size={17} aria-hidden="true" />
          Back to batch results
        </button>
      ) : null}

      <div className="analysis-result-grid">
        <section className="analysis-media-card">
          <header>
            <div>
              <span className="eyebrow">{contextLabel ?? "Analysis result"}</span>
              <h2>{showSource ? "Original image" : "Detected result"}</h2>
            </div>
            <div className="analysis-view-switch" aria-label="Image evidence view">
              <button
                type="button"
                className={showSource ? "" : "active"}
                aria-pressed={!showSource}
                onClick={() => setSourceViewResult(null)}
              >
                Result
              </button>
              <button
                type="button"
                className={showSource ? "active" : ""}
                aria-pressed={showSource}
                onClick={() => setSourceViewResult(result)}
              >
                Original
              </button>
            </div>
          </header>
          <div className="analysis-media-wrap">
            <VideoSurface
              mode="image"
              videoRef={videoRef}
              imageUrl={imageUrl}
              result={showSource ? null : result}
            />
          </div>
        </section>

        <aside className="analysis-summary-card">
          <div className={`analysis-summary-emblem ${primaryEvent ? "detected" : "empty"}`}>
            {primaryEvent ? <BadgeCheck size={45} aria-hidden="true" /> : <ScanSearch size={40} aria-hidden="true" />}
          </div>
          <span className="eyebrow">{primaryEvent ? "Detection" : "Analysis complete"}</span>
          <h2>{headline}</h2>
          {primaryEvent ? (
            meaningUnknown ? (
              <div className="analysis-uncertain-state">
                <strong>Sign detected</strong>
                <span>Meaning uncertain</span>
              </div>
            ) : (
              <div className="analysis-confidence-block">
                <strong>{Math.round(primaryEvent.confidence * 100)}%</strong>
                <span>semantic confidence</span>
              </div>
            )
          ) : null}
          <div className="analysis-summary-divider" />
          <dl className="analysis-summary-facts">
            <div>
              <dt>Advisory</dt>
              <dd>{instruction}</dd>
            </div>
            <div>
              <dt>Target</dt>
              <dd>{primaryEvent ? targetSummary(primaryEvent) : "None"}</dd>
            </div>
            <div>
              <dt>OCR</dt>
              <dd>{primaryEvent?.ocr.text || "No text"}</dd>
            </div>
          </dl>
          <button className="analysis-primary-action full-width" onClick={onChooseImage}>
            <ImagePlus size={18} aria-hidden="true" />
            Analyze another image
          </button>
        </aside>
      </div>

      <details className="analysis-explanation" open>
        <summary>
          <span>
            <span className="eyebrow">Transparent analysis</span>
            <strong>How the result was produced</strong>
          </span>
          <ChevronDown size={20} aria-hidden="true" />
        </summary>
        <div className="analysis-steps">
          <article>
            <span className="analysis-step-number">1</span>
            <ScanSearch size={22} aria-hidden="true" />
            <strong>Locate</strong>
            <p>
              {result.events.length
                ? `${result.events.length} sign candidate${result.events.length === 1 ? "" : "s"} found in the image.`
                : "No reliable sign candidate was retained."}
            </p>
          </article>
          <article>
            <span className="analysis-step-number">2</span>
            <BrainCircuit size={22} aria-hidden="true" />
            <strong>Interpret</strong>
            <p>
              {primaryEvent
                ? meaningUnknown
                  ? "The sign is visible, but its semantic meaning remains uncertain."
                  : `${headline} was selected by the semantic pipeline.`
                : "There was no sign to classify or read."}
            </p>
          </article>
          <article>
            <span className="analysis-step-number">3</span>
            <ShieldCheck size={22} aria-hidden="true" />
            <strong>Advise</strong>
            <p>{instruction}</p>
          </article>
        </div>
      </details>

      <details className="analysis-technical-details">
        <summary>
          <span>
            <span className="eyebrow">For assessment and troubleshooting</span>
            <strong>Technical details</strong>
          </span>
          <ChevronDown size={20} aria-hidden="true" />
        </summary>
        <div className="analysis-technical-content">
          <section className="analysis-technical-metrics" aria-label="Technical metrics">
            <div><Cpu size={17} aria-hidden="true" /><span>Runtime</span><strong>{runtimeLabel}</strong></div>
            <div><Gauge size={17} aria-hidden="true" /><span>Latency</span><strong>{Math.round(result.latency_ms)} ms</strong></div>
            <div><Shapes size={17} aria-hidden="true" /><span>Signs</span><strong>{result.events.length}</strong></div>
          </section>
          <dl className="analysis-runtime-details">
            <div><dt>Pipeline</dt><dd>{result.mode}</dd></div>
            <div><dt>Detector</dt><dd>{detectorRuntime}</dd></div>
            <div><dt>Classifier</dt><dd>{classifierRuntime}</dd></div>
            <div><dt>Detector evidence</dt><dd>{evidenceSummary(primaryEvent, "detector:")}</dd></div>
            <div><dt>Classifier evidence</dt><dd>{evidenceSummary(primaryEvent, "classifier_raw:")}</dd></div>
            <div><dt>Frame</dt><dd>{result.frame_id}</dd></div>
          </dl>
          {modelWarnings.length ? (
            <section className="analysis-warning-panel">
              <TriangleAlert size={18} aria-hidden="true" />
              <div>
                <strong>Development status</strong>
                {modelWarnings.map((warning) => <span key={warning}>{warning}</span>)}
              </div>
            </section>
          ) : null}
        </div>
      </details>
    </section>
  );
}

import { AlertTriangle, Files, ScanLine } from "lucide-react";

import type { FrameResult } from "../types";

export interface BatchDisplayItem {
  filename: string;
  previewUrl: string;
  result?: FrameResult | null;
  error?: string | null;
}

interface BatchResultsProps {
  items: BatchDisplayItem[];
  busy: boolean;
}

export function BatchResults({ items, busy }: BatchResultsProps) {
  if (!items.length) {
    return (
      <section className="batch-empty" aria-label="Batch results">
        <Files size={34} aria-hidden="true" />
        <strong>{busy ? "Analyzing images" : "Select up to 100 images"}</strong>
        <span>Results and detected sign counts will appear here.</span>
      </section>
    );
  }

  return (
    <section className="batch-results" aria-label="Batch results">
      <header>
        <div>
          <span className="eyebrow">Batch analysis</span>
          <h2>{items.length} images</h2>
        </div>
        <span>{busy ? "Processing" : "Complete"}</span>
      </header>
      <div className="batch-table" role="table" aria-label="Batch inference results">
        <div className="batch-row batch-heading" role="row">
          <span role="columnheader">Image</span>
          <span role="columnheader">Signs</span>
          <span role="columnheader">Runtime</span>
          <span role="columnheader">Result</span>
        </div>
        {items.map((item) => {
          const primary = item.result?.events[0];
          return (
            <div className="batch-row" role="row" key={`${item.filename}-${item.previewUrl}`}>
              <div className="batch-file" role="cell">
                <img src={item.previewUrl} alt="" />
                <span title={item.filename}>{item.filename}</span>
              </div>
              <strong role="cell">{item.result?.events.length ?? "—"}</strong>
              <span role="cell">
                {item.result ? `${Math.round(item.result.latency_ms)} ms` : "—"}
              </span>
              <div className={item.error ? "batch-outcome error" : "batch-outcome"} role="cell">
                {item.error ? (
                  <>
                    <AlertTriangle size={14} />
                    <span>{item.error}</span>
                  </>
                ) : (
                  <>
                    <ScanLine size={14} />
                    <span>{primary?.meaning.en ?? "No sign detected"}</span>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

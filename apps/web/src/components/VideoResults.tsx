import { Film, ScanLine } from "lucide-react";

import type { VideoInferenceResponse } from "../types";

interface VideoResultsProps {
  videoUrl: string | null;
  summary: VideoInferenceResponse | null;
  busy: boolean;
}

export function VideoResults({ videoUrl, summary, busy }: VideoResultsProps) {
  if (!videoUrl) {
    return (
      <section className="batch-empty" aria-label="Video analysis">
        <Film size={34} aria-hidden="true" />
        <strong>Select a road video</strong>
        <span>The first 300 frames will be sampled for sign events.</span>
      </section>
    );
  }

  return (
    <section className="video-analysis" aria-label="Video analysis">
      <video src={videoUrl} controls preload="metadata" />
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
          Video processing uses an independent tracking session.
        </p>
      </div>
    </section>
  );
}

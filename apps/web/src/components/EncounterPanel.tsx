import type { EncounterState } from "../encounters";
import type { DisplayLanguage, FrameResult } from "../types";
import { semanticSignName } from "../advisoryDisplay";
import { SignPanel } from "./SignPanel";

export function EncounterPanel({
  state,
  language,
  raw,
}: {
  state: EncounterState;
  language: DisplayLanguage;
  raw?: FrameResult | null;
}) {
  return (
    <div className="encounter-panel">
      {state.primary?.lastSeen && (
        <span className="subtle-note">Last seen</span>
      )}
      <SignPanel event={state.primary?.event ?? null} language={language} />
      {state.checking && (
        <p className="subtle-note" role="status">
          Checking sign…
        </p>
      )}
      {state.active.length > 1 && (
        <ul aria-label="Other confirmed signs">
          {state.active
            .filter((e) => e.id !== state.primary?.id)
            .map((e) => (
              <li key={e.id}>{semanticSignName(e.event, language)}</li>
            ))}
        </ul>
      )}
      {Boolean(raw?.events.length) && (
        <details className="encounter-diagnostics">
          <summary>Detection details</summary>
          {raw?.events.map((e, i) => (
            <div key={`${e.track_id}:${i}`}>
              <strong>{semanticSignName(e, language)}</strong>
              <p>
                Confidence {Math.round(e.confidence * 100)}% · Track{" "}
                {e.track_id}
              </p>
              <p>OCR: {e.ocr.text || "No text"}</p>
              <small>{e.evidence.join(" · ")}</small>
            </div>
          ))}
        </details>
      )}
    </div>
  );
}

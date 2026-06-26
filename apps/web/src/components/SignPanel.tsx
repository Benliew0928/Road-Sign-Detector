import { AlertTriangle, Gauge, Languages, Navigation, ShieldAlert } from "lucide-react";

import type { DisplayLanguage, SignEvent } from "../types";

interface SignPanelProps {
  event: SignEvent | null;
  language: DisplayLanguage;
}

const languageLabel: Record<DisplayLanguage, string> = {
  en: "English",
  ms: "Bahasa Melayu",
  zh: "中文",
};

export function SignPanel({ event, language }: SignPanelProps) {
  if (!event) {
    return (
      <section className="sign-panel empty-panel">
        <ShieldAlert size={28} aria-hidden="true" />
        <div>
          <span className="eyebrow">Current sign</span>
          <h2>No stable sign</h2>
        </div>
      </section>
    );
  }

  const target = event.action.target_speed_kmh;
  return (
    <section className={`sign-panel severity-panel-${event.severity}`}>
      <header className="sign-heading">
        <div>
          <span className="eyebrow">Current sign</span>
          <h2>{event.meaning[language]}</h2>
        </div>
        <span className="confidence">{Math.round(event.confidence * 100)}%</span>
      </header>

      <dl className="sign-facts">
        <div>
          <dt>
            <Navigation size={16} aria-hidden="true" /> ADAS action
          </dt>
          <dd>{event.action.code.replaceAll("_", " ")}</dd>
        </div>
        <div>
          <dt>
            <Gauge size={16} aria-hidden="true" /> Target
          </dt>
          <dd>{target ? `${target} km/h` : "Advisory"}</dd>
        </div>
        <div>
          <dt>
            <Languages size={16} aria-hidden="true" /> OCR
          </dt>
          <dd className="ocr-value">
            <span>{event.ocr.text || "No text"}</span>
            <small>{event.ocr.language}</small>
          </dd>
        </div>
      </dl>

      {!event.stable ? (
        <div className="stability-notice">
          <AlertTriangle size={16} aria-hidden="true" />
          Verifying across frames
        </div>
      ) : (
        <div className="stability-notice stable">
          <span className="status-dot" />
          Stable in {languageLabel[language]}
        </div>
      )}
    </section>
  );
}

import { AlertTriangle, Gauge, Navigation, ShieldAlert } from "lucide-react";
import {
  advisoryInstruction,
  semanticSignName,
  targetSummary,
} from "../advisoryDisplay";
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
  const target = targetSummary(event);
  const directionalActionBlocked = event.evidence.includes(
    "safety:directional_action_blocked",
  );
  return (
    <section className={`sign-panel severity-panel-${event.severity}`}>
      <header className="sign-heading">
        <div>
          <span className="eyebrow">Current sign</span>
          <h2>{semanticSignName(event, language)}</h2>
        </div>
      </header>
      <dl className="sign-facts">
        <div>
          <dt>
            <Navigation size={16} aria-hidden="true" /> ADAS advice
          </dt>
          <dd>{advisoryInstruction(event, language)}</dd>
        </div>
        {target !== "Advisory" && (
          <div>
            <dt>
              <Gauge size={16} aria-hidden="true" /> Target
            </dt>
            <dd>{target}</dd>
          </div>
        )}
      </dl>
      {directionalActionBlocked || event.advisory?.safe_to_announce !== true ? (
        <div className="stability-notice">
          <AlertTriangle size={16} aria-hidden="true" />
          Verify visually: driving guidance unavailable
        </div>
      ) : !event.stable ? (
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

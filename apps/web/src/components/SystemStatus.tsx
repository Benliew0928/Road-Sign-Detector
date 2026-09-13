import {
  Cpu,
  RefreshCw,
  ScanLine,
  Type,
  Sparkles,
  ChevronUp,
} from "lucide-react";
import type { HealthResponse } from "../types";
export function SystemStatus({
  health,
  refresh,
}: {
  health: HealthResponse | null;
  refresh: () => void;
}) {
  const m = health?.models;
  return (
    <details className="system-details compact-insight">
      <summary aria-label="System details">
        <Cpu size={16} />
        <span>System</span>
        <i className={`insight-dot ${m ? "ready" : ""}`} />
      </summary>
      <div className="insight-popover system-popover">
        <div className="insight-heading">
          <span className="insight-orbit">
            <Cpu size={22} />
          </span>
          <div>
            <small>LOCAL ENGINE</small>
            <h3>{m ? "Ready to explore" : "Engine offline"}</h3>
          </div>
          <button
            onClick={refresh}
            aria-label="Refresh status"
            title="Refresh status"
          >
            <RefreshCw size={16} />
          </button>
        </div>
        <div className="insight-chips">
          <span>
            {m?.mode === "deep"
              ? "Semantic AI"
              : m?.mode === "baseline"
                ? "Classical baseline"
                : m
                  ? "Auto pipeline"
                  : "Disconnected"}
          </span>
          {m && <span>{m.detector_device || "Default device"}</span>}
        </div>
        {m && (
          <div className="capability-list">
            {[
              {
                label: "Detect signs",
                ready: m.detector_loaded,
                icon: ScanLine,
              },
              {
                label: "Understand signs",
                ready: m.classifier_loaded,
                icon: Sparkles,
              },
              { label: "Read text", ready: m.ocr_loaded, icon: Type },
            ].map(({ label, ready, icon: Icon }) => (
              <div key={label}>
                <Icon size={16} />
                <span>{label}</span>
                <strong className={ready ? "available" : "unavailable"}>
                  {ready ? "Ready" : "Unavailable"}
                </strong>
              </div>
            ))}
          </div>
        )}
        <div className="insight-foot">
          <span>
            {m?.warnings.length
              ? `${m.warnings.length} engine ${m.warnings.length === 1 ? "notice" : "notices"}`
              : "On this device"}
          </span>
          <ChevronUp size={14} />
        </div>
      </div>
    </details>
  );
}

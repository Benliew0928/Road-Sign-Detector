import type { Encounter } from "../encounters";
import { actionSignature } from "../encounters";
import type { FrameResult, SignEvent } from "../types";
export function sign(overrides: Partial<SignEvent> = {}): SignEvent {
  return {
    schema_version: "1",
    frame_id: 0,
    track_id: 1,
    coursework_id: null,
    semantic_sign_id: "stop",
    meaning: { en: "Stop", ms: "Berhenti", zh: "Stop" },
    ocr: {
      text: "",
      confidence: 0,
      script: "none",
      language: "unknown",
      numeric_value: null,
      unit: null,
      semantic_sign_id: null,
    },
    confidence: 0.95,
    bbox: { x1: 10, y1: 10, x2: 100, y2: 100 },
    mask: null,
    action: {
      code: "STOP",
      target_speed_kmh: null,
      restriction_value: null,
      restriction_unit: null,
      direction: null,
      advisory_only: true,
    },
    advisory: {
      headline: { en: "Stop", ms: "Stop", zh: "Stop" },
      instruction: {
        en: "Stop safely.",
        ms: "Stop safely.",
        zh: "Stop safely.",
      },
      safe_to_announce: true,
    },
    severity: "critical",
    latency_ms: 1,
    device: "test",
    stable: true,
    should_announce: false,
    evidence: [],
    ...overrides,
  };
}
export function frame(id: number, events: SignEvent[] = [sign()]): FrameResult {
  return {
    frame_id: id,
    width: 640,
    height: 480,
    mode: "deep",
    latency_ms: 1,
    events,
    warnings: [],
  };
}
export function encounter(id = "one", at = 0, event = sign()): Encounter {
  return {
    id,
    event,
    signature: actionSignature(event),
    revision: 1,
    confirmedAt: at,
    lastSupportedAt: at,
    lastSeen: false,
  };
}

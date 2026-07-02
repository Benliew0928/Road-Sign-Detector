import { describe, expect, it } from "vitest";

import {
  advisoryHeadline,
  advisoryInstruction,
  formatActionCode,
  targetSummary,
} from "./advisoryDisplay";
import type { SignEvent } from "./types";

function event(overrides: Partial<SignEvent> = {}): SignEvent {
  return {
    schema_version: "1.0",
    frame_id: 1,
    track_id: 2,
    coursework_id: null,
    semantic_sign_id: "maximum_speed",
    meaning: { en: "Maximum speed", ms: "Had laju maksimum", zh: "Maximum speed" },
    ocr: {
      text: "50",
      confidence: 0.99,
      script: "latin",
      language: "en",
      numeric_value: 50,
      unit: "KM/H",
      semantic_sign_id: null,
    },
    confidence: 0.96,
    bbox: { x1: 0, y1: 0, x2: 10, y2: 10 },
    mask: null,
    action: {
      code: "SET_TARGET_SPEED",
      target_speed_kmh: 50,
      restriction_value: null,
      restriction_unit: null,
      direction: null,
      advisory_only: true,
    },
    advisory: {
      headline: {
        en: "Speed limit 50 km/h",
        ms: "Had laju 50 km/j",
        zh: "Speed limit 50 km/h",
      },
      instruction: {
        en: "This road has a speed limit of 50 km/h. Keep your speed at or below the limit.",
        ms: "Jalan ini mempunyai had laju 50 km/j. Pastikan kelajuan tidak melebihi had.",
        zh: "This road has a speed limit of 50 km/h. Keep your speed at or below the limit.",
      },
      safe_to_announce: true,
    },
    severity: "critical",
    latency_ms: 10,
    device: "test",
    stable: true,
    should_announce: true,
    evidence: [],
    ...overrides,
  };
}

describe("advisory display helpers", () => {
  it("prefers backend advisory text over raw action names", () => {
    const sign = event();

    expect(advisoryHeadline(sign, "en")).toBe("Speed limit 50 km/h");
    expect(advisoryInstruction(sign, "en")).toContain("Keep your speed");
    expect(targetSummary(sign)).toBe("50 km/h");
  });

  it("falls back safely for older events without advisory text", () => {
    const sign = event({ advisory: undefined });

    expect(advisoryHeadline(sign, "en")).toBe("Maximum speed");
    expect(advisoryInstruction(sign, "en")).toBe("Set Target Speed");
    expect(formatActionCode("UNKNOWN_CAUTION")).toBe("Unknown Caution");
  });
});

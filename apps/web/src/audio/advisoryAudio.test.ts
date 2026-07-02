import { describe, expect, it } from "vitest";

import {
  chooseAdvisoryEvent,
  resolveAdvisoryPhraseId,
  type AdvisoryAudioManifest,
  type AdvisoryAudioPhrase,
} from "./advisoryAudio";
import type { SignEvent } from "../types";

function phrase(phrase_id: string, priority: number): AdvisoryAudioPhrase {
  return {
    phrase_id,
    semantic_sign_id: phrase_id,
    audio_key: phrase_id,
    base_action: "UNKNOWN_CAUTION",
    severity: priority >= 4 ? "critical" : priority >= 3 ? "warning" : "caution",
    priority,
    interrupts_lower_priority: priority >= 3,
    cooldown_seconds: 8,
    parameter: null,
    text: { en: phrase_id, ms: phrase_id, zh: phrase_id },
    assets: {
      en: { src: `/audio/${phrase_id}.wav`, sha256: null, bytes: null, duration_seconds: null, voice: null, generated: true },
      ms: { src: `/audio/${phrase_id}.wav`, sha256: null, bytes: null, duration_seconds: null, voice: null, generated: true },
      zh: { src: `/audio/${phrase_id}.wav`, sha256: null, bytes: null, duration_seconds: null, voice: null, generated: true },
    },
  };
}

const manifest: AdvisoryAudioManifest = {
  schema_version: "test",
  catalogue_version: "test",
  languages: ["en", "ms", "zh"],
  description: "test",
  fallback_phrase_id: "unknown_sign",
  semantic_phrase_ids: {
    maximum_speed: "maximum_speed",
    parking: "parking",
  },
  audio_key_phrase_ids: {},
  variant_phrase_ids: {
    speed_limit_kmh: { "50": "speed_limit_50_kmh" },
    minimum_speed_kmh: {},
    temporary_speed_limit_kmh: {},
    height_limit_m: { "4.5": "height_limit_4_5_m" },
    width_limit_m: {},
    weight_limit_t: {},
  },
  phrases: {
    maximum_speed: phrase("maximum_speed", 4),
    speed_limit_50_kmh: phrase("speed_limit_50_kmh", 4),
    height_limit_4_5_m: phrase("height_limit_4_5_m", 4),
    parking: phrase("parking", 1),
    unknown_sign: phrase("unknown_sign", 2),
  },
};

function event(overrides: Partial<SignEvent>): SignEvent {
  return {
    schema_version: "1.0",
    frame_id: 1,
    track_id: 2,
    coursework_id: null,
    semantic_sign_id: "unknown_sign",
    meaning: { en: "Unknown", ms: "Tidak dikenali", zh: "未知" },
    ocr: {
      text: "",
      confidence: 0,
      script: "none",
      language: "unknown",
      numeric_value: null,
      unit: null,
      semantic_sign_id: null,
    },
    confidence: 0.9,
    bbox: { x1: 0, y1: 0, x2: 10, y2: 10 },
    mask: null,
    action: {
      code: "UNKNOWN_CAUTION",
      target_speed_kmh: null,
      restriction_value: null,
      restriction_unit: null,
      direction: null,
      advisory_only: true,
    },
    severity: "caution",
    latency_ms: 10,
    device: "test",
    stable: true,
    should_announce: true,
    evidence: [],
    ...overrides,
  };
}

describe("advisory audio resolver", () => {
  it("uses parameterized speed warnings when speed is known", () => {
    const selected = resolveAdvisoryPhraseId(
      event({
        semantic_sign_id: "maximum_speed",
        action: {
          code: "SET_TARGET_SPEED",
          target_speed_kmh: 50,
          restriction_value: null,
          restriction_unit: null,
          direction: null,
          advisory_only: true,
        },
        severity: "critical",
      }),
      manifest,
    );

    expect(selected).toBe("speed_limit_50_kmh");
  });

  it("uses parameterized restriction warnings when a known restriction is present", () => {
    const selected = resolveAdvisoryPhraseId(
      event({
        semantic_sign_id: "height_restriction",
        action: {
          code: "HEIGHT_RESTRICTION",
          target_speed_kmh: null,
          restriction_value: 4.5,
          restriction_unit: "M",
          direction: null,
          advisory_only: true,
        },
        severity: "critical",
      }),
      manifest,
    );

    expect(selected).toBe("height_limit_4_5_m");
  });

  it("falls back safely for unknown signs", () => {
    expect(resolveAdvisoryPhraseId(event({ semantic_sign_id: "not_in_manifest" }), manifest)).toBe(
      "unknown_sign",
    );
  });

  it("chooses the highest priority announceable event", () => {
    const selected = chooseAdvisoryEvent(
      [
        event({ semantic_sign_id: "parking", severity: "information", confidence: 0.99 }),
        event({ semantic_sign_id: "maximum_speed", severity: "critical", confidence: 0.8 }),
      ],
      manifest,
    );

    expect(selected?.phraseId).toBe("maximum_speed");
  });

  it("does not announce events marked unsafe by the backend advisory", () => {
    const selected = chooseAdvisoryEvent(
      [
        event({
          semantic_sign_id: "maximum_speed",
          severity: "critical",
          advisory: {
            headline: { en: "Maximum speed", ms: "Maximum speed", zh: "Maximum speed" },
            instruction: {
              en: "This sign is not confident enough for a strong command.",
              ms: "This sign is not confident enough for a strong command.",
              zh: "This sign is not confident enough for a strong command.",
            },
            safe_to_announce: false,
          },
        }),
      ],
      manifest,
    );

    expect(selected).toBeNull();
  });
});

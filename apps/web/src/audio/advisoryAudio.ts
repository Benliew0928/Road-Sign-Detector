import { safeForSpeech } from "../encounters";
import type { DisplayLanguage, SignEvent } from "../types";
export interface AdvisoryAudioAsset {
  src: string;
  fallback_src?: string | null;
  sha256: string | null;
  bytes: number | null;
  duration_seconds: number | null;
  voice: string | null;
  generated: boolean;
  provider?: string | null;
  model?: string | null;
  style_profile?: string | null;
}
export interface AdvisoryAudioPhrase {
  phrase_id: string;
  semantic_sign_id: string | null;
  audio_key: string | null;
  base_action: string;
  severity: SignEvent["severity"];
  priority: number;
  interrupts_lower_priority: boolean;
  cooldown_seconds: number;
  parameter: { kind: string; value: number; unit: string } | null;
  text: Record<DisplayLanguage, string>;
  assets: Record<DisplayLanguage, AdvisoryAudioAsset>;
}
export interface AdvisoryAudioManifest {
  schema_version: string;
  catalogue_version: string;
  languages: DisplayLanguage[];
  description: string;
  audio_pack?: {
    kind: string;
    status: string;
    provider: string;
    model: string;
    voices: Record<DisplayLanguage, string>;
    style_profile: string;
    style_label: string;
    generated_at: string;
    runtime_policy: string;
    fallback_audio_root: string;
    selected_phrase_ids: string[] | null;
  };
  fallback_phrase_id: string;
  semantic_phrase_ids: Record<string, string>;
  audio_key_phrase_ids: Record<string, string>;
  variant_phrase_ids: {
    speed_limit_kmh: Record<string, string>;
    minimum_speed_kmh: Record<string, string>;
    temporary_speed_limit_kmh: Record<string, string>;
    height_limit_m: Record<string, string>;
    width_limit_m: Record<string, string>;
    weight_limit_t: Record<string, string>;
  };
  phrases: Record<string, AdvisoryAudioPhrase>;
}
const SEVERITY_PRIORITY: Record<SignEvent["severity"], number> = {
  information: 1,
  caution: 2,
  warning: 3,
  critical: 4,
};
function normalizedValue(value: number): string {
  return String(value);
}
function exactVariant(
  variants: Record<string, string>,
  value: number | null,
): string | null {
  if (value === null) return null;
  return variants[normalizedValue(value)] ?? null;
}
export function resolveAdvisoryPhraseId(
  event: SignEvent,
  manifest: AdvisoryAudioManifest,
): string {
  if (event.action.target_speed_kmh !== null) {
    const speed = event.action.target_speed_kmh;
    if (event.semantic_sign_id === "minimum_speed") {
      const phraseId = exactVariant(
        manifest.variant_phrase_ids.minimum_speed_kmh,
        speed,
      );
      if (phraseId) return phraseId;
    }
    if (event.semantic_sign_id === "temporary_speed_limit") {
      const phraseId = exactVariant(
        manifest.variant_phrase_ids.temporary_speed_limit_kmh,
        speed,
      );
      if (phraseId) return phraseId;
    }
    const phraseId =
      event.semantic_sign_id === "maximum_speed"
        ? exactVariant(manifest.variant_phrase_ids.speed_limit_kmh, speed)
        : null;
    if (phraseId) return phraseId;
  }
  if (event.action.restriction_value !== null) {
    const unit = event.action.restriction_unit?.toUpperCase() ?? "";
    if (event.action.code === "HEIGHT_RESTRICTION" && unit === "M") {
      const phraseId = exactVariant(
        manifest.variant_phrase_ids.height_limit_m,
        event.action.restriction_value,
      );
      if (phraseId) return phraseId;
    }
    if (event.action.code === "WIDTH_RESTRICTION" && unit === "M") {
      const phraseId = exactVariant(
        manifest.variant_phrase_ids.width_limit_m,
        event.action.restriction_value,
      );
      if (phraseId) return phraseId;
    }
    if (event.action.code === "WEIGHT_RESTRICTION" && unit === "T") {
      const phraseId = exactVariant(
        manifest.variant_phrase_ids.weight_limit_t,
        event.action.restriction_value,
      );
      if (phraseId) return phraseId;
    }
  }
  return (
    manifest.semantic_phrase_ids[event.semantic_sign_id] ??
    manifest.audio_key_phrase_ids[event.semantic_sign_id] ??
    ""
  );
}
export function advisoryEventPriority(
  event: SignEvent,
  phrase: AdvisoryAudioPhrase | undefined,
): number {
  return phrase?.priority ?? SEVERITY_PRIORITY[event.severity] ?? 2;
}
export function chooseAdvisoryEvent(
  events: SignEvent[],
  manifest: AdvisoryAudioManifest,
): { event: SignEvent; phrase: AdvisoryAudioPhrase; phraseId: string } | null {
  const candidates = events
    .filter((event) => event.should_announce && safeForSpeech(event))
    .map((event) => {
      const phraseId = resolveAdvisoryPhraseId(event, manifest);
      const phrase = manifest.phrases[phraseId];
      return phrase ? { event, phrase, phraseId: phrase.phrase_id } : null;
    })
    .filter(
      (
        item,
      ): item is {
        event: SignEvent;
        phrase: AdvisoryAudioPhrase;
        phraseId: string;
      } => Boolean(item),
    );
  candidates.sort((first, second) => {
    const priorityDelta =
      advisoryEventPriority(second.event, second.phrase) -
      advisoryEventPriority(first.event, first.phrase);
    if (priorityDelta !== 0) return priorityDelta;
    return second.event.confidence - first.event.confidence;
  });
  return candidates[0] ?? null;
}

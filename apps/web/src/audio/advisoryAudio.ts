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
  return Number.isInteger(value)
    ? String(value)
    : String(Number(value.toFixed(2))).replace(/\.0$/, "");
}

function closestVariant(
  variants: Record<string, string>,
  value: number | null,
  tolerance: number,
): string | null {
  if (value === null) return null;
  const exact = variants[normalizedValue(value)];
  if (exact) return exact;
  let bestDistance = Number.POSITIVE_INFINITY;
  let bestPhraseId: string | null = null;
  for (const [candidate, phraseId] of Object.entries(variants)) {
    const numeric = Number(candidate);
    if (!Number.isFinite(numeric)) continue;
    const distance = Math.abs(numeric - value);
    if (distance <= tolerance && distance < bestDistance) {
      bestDistance = distance;
      bestPhraseId = phraseId;
    }
  }
  return bestPhraseId;
}

export function resolveAdvisoryPhraseId(
  event: SignEvent,
  manifest: AdvisoryAudioManifest,
): string {
  if (event.action.target_speed_kmh !== null) {
    const speed = Math.round(event.action.target_speed_kmh);
    if (event.semantic_sign_id === "minimum_speed") {
      const phraseId = closestVariant(manifest.variant_phrase_ids.minimum_speed_kmh, speed, 2);
      if (phraseId) return phraseId;
    }
    if (event.semantic_sign_id === "temporary_speed_limit") {
      const phraseId = closestVariant(
        manifest.variant_phrase_ids.temporary_speed_limit_kmh,
        speed,
        2,
      );
      if (phraseId) return phraseId;
    }
    const phraseId = closestVariant(manifest.variant_phrase_ids.speed_limit_kmh, speed, 2);
    if (phraseId) return phraseId;
  }

  if (event.action.restriction_value !== null) {
    const unit = event.action.restriction_unit?.toUpperCase() ?? "";
    if (event.action.code === "HEIGHT_RESTRICTION" && unit === "M") {
      const phraseId = closestVariant(
        manifest.variant_phrase_ids.height_limit_m,
        event.action.restriction_value,
        0.15,
      );
      if (phraseId) return phraseId;
    }
    if (event.action.code === "WIDTH_RESTRICTION" && unit === "M") {
      const phraseId = closestVariant(
        manifest.variant_phrase_ids.width_limit_m,
        event.action.restriction_value,
        0.15,
      );
      if (phraseId) return phraseId;
    }
    if (event.action.code === "WEIGHT_RESTRICTION" && unit === "T") {
      const phraseId = closestVariant(
        manifest.variant_phrase_ids.weight_limit_t,
        event.action.restriction_value,
        0.5,
      );
      if (phraseId) return phraseId;
    }
  }

  return (
    manifest.semantic_phrase_ids[event.semantic_sign_id] ??
    manifest.audio_key_phrase_ids[event.semantic_sign_id] ??
    manifest.fallback_phrase_id
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
    .filter((event) => event.should_announce && (event.advisory?.safe_to_announce ?? true))
    .map((event) => {
      const phraseId = resolveAdvisoryPhraseId(event, manifest);
      const phrase = manifest.phrases[phraseId] ?? manifest.phrases[manifest.fallback_phrase_id];
      return phrase ? { event, phrase, phraseId: phrase.phrase_id } : null;
    })
    .filter((item): item is { event: SignEvent; phrase: AdvisoryAudioPhrase; phraseId: string } =>
      Boolean(item),
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

import type { DisplayLanguage, LocalizedMeaning, SignEvent } from "./types";

export function localizedText(
  value: LocalizedMeaning | undefined,
  language: DisplayLanguage,
  fallback = "",
): string {
  return value?.[language] || value?.en || fallback;
}

export function formatActionCode(code: string): string {
  return code
    .toLowerCase()
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function advisoryHeadline(event: SignEvent, language: DisplayLanguage): string {
  return localizedText(event.advisory?.headline, language, localizedText(event.meaning, language));
}

export function advisoryInstruction(event: SignEvent, language: DisplayLanguage): string {
  return localizedText(
    event.advisory?.instruction,
    language,
    formatActionCode(event.action.code),
  );
}

export function targetSummary(event: SignEvent): string {
  if (event.action.target_speed_kmh !== null) {
    return `${event.action.target_speed_kmh} km/h`;
  }
  if (event.action.restriction_value !== null && event.action.restriction_unit) {
    return `${event.action.restriction_value} ${event.action.restriction_unit.toLowerCase()}`;
  }
  return "Advisory";
}

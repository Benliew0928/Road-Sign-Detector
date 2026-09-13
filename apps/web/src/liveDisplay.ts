import type { Encounter, EncounterState } from "./encounters";

export const LIVE_DISPLAY_HOLD_MS = 5000;
const FRESH_MS = 1000;
export interface LiveDisplay {
  source: string;
  shownAt: number;
  sign: Encounter | null;
}
// Presentation only: held snapshots never alter audio eligibility or tracking.
export function advanceLiveDisplay(
  previous: LiveDisplay,
  state: EncounterState,
): LiveDisplay {
  const current =
    previous.source === state.source && state.at >= previous.shownAt
      ? previous
      : { source: state.source, shownAt: state.at, sign: null };
  if (state.frame < 0) return { source: state.source, shownAt: state.at, sign: null };
  const candidates = state.active.filter(
    (e) =>
      e.event.semantic_sign_id !== "unknown_sign" &&
      e.event.confidence >= 0.8 &&
      state.at - e.lastSupportedAt < FRESH_MS,
  );
  candidates.sort(
    (a, b) =>
      b.event.confidence - a.event.confidence ||
      Number(b.signature === current.sign?.signature) -
        Number(a.signature === current.sign?.signature),
  );
  const matching = candidates.find(
    (e) => e.signature === current.sign?.signature,
  );
  if (current.sign && state.at - current.shownAt < LIVE_DISPLAY_HOLD_MS) {
    return {
      ...current,
      sign: matching ?? {
        ...current.sign,
        lastSeen: true,
        speechEligible: false,
      },
    };
  }
  const next = candidates[0] ?? null;
  if (next && next.signature === current.sign?.signature)
    return { ...current, sign: next };
  return { source: state.source, shownAt: state.at, sign: next };
}

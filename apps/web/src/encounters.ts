import type {
  BoundingBox,
  FrameResult,
  SignEvent,
  VideoInferenceResponse,
} from "./types";

export const ENCOUNTER_POLICY = {
  window: 1500,
  bucket: 100,
  span: 300,
  hold: 2000,
  bridge: 1000,
  rearm: 5000,
  history: 40,
};
export const AUDIO_POLICY = { queueLimit: 3, gap: 1000, expiry: 5000 };
export const severityRank = {
  information: 1,
  caution: 2,
  warning: 3,
  critical: 4,
};
export interface Encounter {
  id: string;
  signature: string;
  event: SignEvent;
  revision: number;
  confirmedAt: number;
  lastSupportedAt: number;
  lastSeen: boolean;
  speechEligible?: boolean;
}
interface Observation {
  at: number;
  signature: string;
  event: SignEvent;
}
interface Track {
  id: string;
  trackId: number;
  bbox: BoundingBox;
  seen: number;
  samples: Observation[];
  confirmed?: Encounter;
}
export interface EncounterState {
  source: string;
  at: number;
  frame: number;
  nextId: number;
  tracks: Track[];
  active: Encounter[];
  history: Encounter[];
  primary: Encounter | null;
  checking: boolean;
}
export function emptyEncounters(source: string): EncounterState {
  return {
    source,
    at: 0,
    frame: -1,
    nextId: 1,
    tracks: [],
    active: [],
    history: [],
    primary: null,
    checking: false,
  };
}
export function actionSignature(e: SignEvent): string {
  return JSON.stringify([
    e.semantic_sign_id,
    e.action.code,
    e.action.direction,
    e.action.target_speed_kmh,
    e.action.restriction_value,
    e.action.restriction_unit,
  ]);
}
export function safeForSpeech(e: SignEvent): boolean {
  return (
    e.semantic_sign_id !== "unknown_sign" &&
    e.action.code !== "UNKNOWN_CAUTION" &&
    e.advisory?.safe_to_announce === true &&
    !e.evidence.includes("safety:directional_action_blocked")
  );
}
function iou(a: BoundingBox, b: BoundingBox): number {
  const intersection =
    Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)) *
    Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  return (
    intersection /
    Math.max(
      1,
      (a.x2 - a.x1) * (a.y2 - a.y1) +
        (b.x2 - b.x1) * (b.y2 - b.y1) -
        intersection,
    )
  );
}
export function advanceEncounters(
  previous: EncounterState,
  at: number,
  result?: FrameResult,
): EncounterState {
  if (at < previous.at || (result && result.frame_id <= previous.frame))
    return previous;
  const state = {
    ...previous,
    at,
    tracks: previous.tracks.map((t) => ({ ...t, samples: [...t.samples] })),
    history: [...previous.history],
  };
  const seen = new Set<string>();
  if (result) {
    state.frame = result.frame_id;
    state.checking = result.events.some(
      (e) => e.semantic_sign_id === "unknown_sign" || !e.stable,
    );
    const fresh = result.events.filter(
      (e) => !e.evidence.some((x) => x.startsWith("tracker_hold:")),
    );
    for (const event of fresh) {
      let track = state.tracks.find(
        (t) => t.trackId === event.track_id && !seen.has(t.id),
      );
      if (!track) {
        const bridges = state.tracks.filter(
          (t) =>
            !seen.has(t.id) &&
            !fresh.some((e) => e.track_id === t.trackId) &&
            at - t.seen <= ENCOUNTER_POLICY.bridge &&
            (t.confirmed?.event.semantic_sign_id ??
              t.samples.at(-1)?.event.semantic_sign_id) ===
              event.semantic_sign_id &&
            iou(t.bbox, event.bbox) >= 0.3,
        );
        if (
          bridges.length === 1 &&
          fresh.filter(
            (e) =>
              e.semantic_sign_id === event.semantic_sign_id &&
              iou(bridges[0].bbox, e.bbox) >= 0.3,
          ).length === 1
        )
          track = bridges[0];
      }
      if (!track) {
        track = {
          id: `${state.source}:${state.nextId++}`,
          trackId: event.track_id,
          bbox: event.bbox,
          seen: at,
          samples: [],
        };
        state.tracks.push(track);
      }
      seen.add(track.id);
      track.trackId = event.track_id;
      track.bbox = event.bbox;
      track.seen = at;
      const signature = actionSignature(event);
      const eligible =
        event.stable &&
        event.semantic_sign_id !== "unknown_sign" &&
        event.confidence >= (event.severity === "critical" ? 0.88 : 0.75);
      track.samples = track.samples.filter(
        (s) => at - s.at <= ENCOUNTER_POLICY.window,
      );
      if (
        !track.samples.length ||
        at - track.samples[track.samples.length - 1].at >=
          ENCOUNTER_POLICY.bucket
      ) {
        track.samples.push({ at, signature: eligible ? signature : "", event });
      }
      const agreeing = track.samples.filter(
        (s) => s.signature === signature && eligible,
      );
      if (
        agreeing.length >= 3 &&
        agreeing.length / track.samples.length >= 0.8 &&
        at - agreeing[0].at >= ENCOUNTER_POLICY.span
      ) {
        if (track.confirmed?.signature !== signature) {
          track.confirmed = {
            id: track.id,
            signature,
            event,
            revision: (track.confirmed?.revision ?? 0) + 1,
            confirmedAt: at,
            lastSupportedAt: at,
            lastSeen: false,
          };
        }
      }
      if (track.confirmed)
        track.confirmed = {
          ...track.confirmed,
          speechEligible:
            eligible &&
            track.confirmed.signature === signature &&
            safeForSpeech(event),
        };
      if (
        track.confirmed &&
        eligible &&
        track.confirmed.signature === signature
      ) {
        // Preserve the confirmed presentation snapshot; live diagnostics remain in the raw result.
        track.confirmed = {
          ...track.confirmed,
          lastSupportedAt: at,
          lastSeen: false,
          event: track.confirmed.event,
        };
      }
    }
  }
  state.active = [];
  for (const track of state.tracks) {
    if (!track.confirmed) continue;
    const encounter = {
      ...track.confirmed,
      lastSeen: at - track.confirmed.lastSupportedAt >= ENCOUNTER_POLICY.bucket,
    };
    track.confirmed = encounter;
    state.history = state.history.filter((e) => e.id !== encounter.id);
    state.history.push(encounter);
    if (at - encounter.lastSupportedAt <= ENCOUNTER_POLICY.hold)
      state.active.push(encounter);
  }
  state.history.sort((a, b) => b.confirmedAt - a.confirmedAt);
  state.history = state.history.slice(0, ENCOUNTER_POLICY.history);
  state.tracks = state.tracks.filter(
    (t) => at - t.seen <= ENCOUNTER_POLICY.rearm,
  );
  state.active.sort(
    (a, b) =>
      severityRank[b.event.severity] - severityRank[a.event.severity] ||
      a.confirmedAt - b.confirmedAt,
  );
  const current = state.active.find((e) => e.id === previous.primary?.id);
  state.primary =
    current &&
    (!state.active[0] ||
      severityRank[current.event.severity] >=
        severityRank[state.active[0].event.severity])
      ? current
      : (state.active[0] ?? null);
  if (!result && !state.tracks.some((t) => at - t.seen < ENCOUNTER_POLICY.hold))
    state.checking = false;
  return state;
}
export function buildEncounterTimeline(
  summary: VideoInferenceResponse | null,
  source: string,
) {
  let state = emptyEncounters(source);
  const moments = new Map<string, Encounter>();
  const timeline = [...(summary?.frame_results ?? [])]
    .sort((a, b) => a.source_frame - b.source_frame)
    .map((frame) => {
      state = advanceEncounters(
        state,
        (frame.source_frame / (summary?.fps || 30)) * 1000,
        frame.result,
      );
      state.history.forEach((e) => moments.set(e.id, e));
      return { at: state.at, state };
    });
  return {
    timeline,
    moments: [...moments.values()].sort(
      (a, b) => a.confirmedAt - b.confirmedAt,
    ),
  };
}
export function encounterAt(
  timeline: ReturnType<typeof buildEncounterTimeline>["timeline"],
  at: number,
  source: string,
): EncounterState {
  let lo = 0,
    hi = timeline.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (timeline[mid].at <= at) lo = mid + 1;
    else hi = mid;
  }
  return lo
    ? advanceEncounters(timeline[lo - 1].state, at)
    : emptyEncounters(source);
}

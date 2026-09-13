import { describe, it, expect } from "vitest";
import {
  advanceEncounters,
  emptyEncounters,
  buildEncounterTimeline,
  encounterAt,
} from "./encounters";
import { frame, sign } from "./test/encounterFixtures";

describe("confirmed encounters", () => {
  it.each([10, 30, 60])(
    "confirms once over hundreds of observations at %i fps",
    (fps) => {
      let state = emptyEncounters("a");
      for (let i = 0; i < fps * 20; i++)
        state = advanceEncounters(state, (i / fps) * 1000, frame(i));
      expect(state.history).toHaveLength(1);
      expect(state.primary?.revision).toBe(1);
      expect(state.primary?.confirmedAt).toBeGreaterThanOrEqual(300);
      expect(state.primary?.confirmedAt).toBeLessThanOrEqual(600);
    },
  );
  it("unknown-only and held boxes never confirm", () => {
    for (const event of [
      sign({ semantic_sign_id: "unknown_sign" }),
      sign({ evidence: ["tracker_hold:1"] }),
    ]) {
      let state = emptyEncounters("a");
      for (let i = 0; i < 100; i++)
        state = advanceEncounters(state, i * 100, frame(i, [event]));
      expect(state.history).toHaveLength(0);
    }
  });
  it("keeps the snapshot through unknowns and expires without frames", () => {
    let state = emptyEncounters("a");
    for (let i = 0; i < 5; i++)
      state = advanceEncounters(state, i * 100, frame(i));
    state = advanceEncounters(
      state,
      500,
      frame(5, [sign({ semantic_sign_id: "unknown_sign" })]),
    );
    expect(state.primary?.event.semantic_sign_id).toBe("stop");
    expect(state.primary?.lastSeen).toBe(true);
    expect(state.primary?.speechEligible).toBe(false);
    state = advanceEncounters(state, 2500);
    expect(state.primary).toBeNull();
    expect(state.history).toHaveLength(1);
  });
  it("bridges a unique changed track, but keeps separate boxes separate", () => {
    let state = emptyEncounters("a");
    for (let i = 0; i < 5; i++)
      state = advanceEncounters(state, i * 100, frame(i));
    for (let i = 5; i < 10; i++)
      state = advanceEncounters(
        state,
        i * 100,
        frame(i, [sign({ track_id: 2 })]),
      );
    expect(state.history).toHaveLength(1);
    for (let i = 10; i < 16; i++)
      state = advanceEncounters(
        state,
        i * 100,
        frame(i, [
          sign({ track_id: 2 }),
          sign({ track_id: 3, bbox: { x1: 200, y1: 200, x2: 300, y2: 300 } }),
        ]),
      );
    expect(state.active).toHaveLength(2);
  });
  it("does not confirm rapidly alternating labels; confirms a sustained revision", () => {
    let state = emptyEncounters("a");
    for (let i = 0; i < 30; i++)
      state = advanceEncounters(
        state,
        i * 100,
        frame(i, [sign({ semantic_sign_id: i % 2 ? "parking" : "stop" })]),
      );
    expect(state.history).toHaveLength(0);
    for (let i = 30; i < 50; i++)
      state = advanceEncounters(state, i * 100, frame(i));
    for (let i = 50; i < 70; i++)
      state = advanceEncounters(
        state,
        i * 100,
        frame(i, [
          sign({ semantic_sign_id: "parking", severity: "information" }),
        ]),
      );
    expect(state.history).toHaveLength(1);
    expect(state.primary?.revision).toBe(2);
  });
  it("ignores duplicate/out-of-order frames and prioritizes a confirmed critical sign", () => {
    let state = emptyEncounters("a");
    for (let i = 0; i < 5; i++)
      state = advanceEncounters(
        state,
        i * 100,
        frame(i, [sign({ severity: "information" })]),
      );
    expect(advanceEncounters(state, 600, frame(3))).toBe(state);
    for (let i = 5; i < 10; i++)
      state = advanceEncounters(
        state,
        i * 100,
        frame(i, [
          sign({ severity: "information" }),
          sign({ track_id: 2, bbox: { x1: 200, y1: 10, x2: 300, y2: 100 } }),
        ]),
      );
    expect(state.primary?.event.track_id).toBe(2);
  });
  it("uses source time and returns the same snapshot when seeking", () => {
    const summary = {
      frames_read: 100,
      sampled_frames: 100,
      events: 100,
      fps: 10,
      event_samples: [],
      representative_result: null,
      frame_results: Array.from({ length: 100 }, (_, i) => ({
        source_frame: i,
        result: frame(i),
      })),
    };
    const { timeline, moments } = buildEncounterTimeline(summary, "video");
    expect(moments).toHaveLength(1);
    expect(encounterAt(timeline, 200, "video").primary).toBeNull();
    expect(encounterAt(timeline, 9000, "video").primary?.id).toBe(
      encounterAt(timeline, 1000, "video").primary?.id,
    );
  });
});

import { describe, it, expect } from "vitest";
import { advanceLiveDisplay, type LiveDisplay } from "./liveDisplay";
import { emptyEncounters, type Encounter } from "./encounters";
import { sign } from "./test/encounterFixtures";
const a: Encounter = {
  id: "a",
  signature: "stop",
  event: sign({ confidence: 0.9 }),
  revision: 1,
  confirmedAt: 0,
  lastSupportedAt: 0,
  lastSeen: false,
};
const b: Encounter = {
  ...a,
  id: "b",
  signature: "other",
  event: sign({ confidence: 0.98 }),
};
const state = (at: number, active: Encounter[]) => ({
  ...emptyEncounters("camera"),
  frame: 1,
  at,
  active,
});
const initial: LiveDisplay = { source: "camera", shownAt: 0, sign: null };
describe("live guidance dwell", () => {
  it("holds through noise and stronger competitors for a full five seconds", () => {
    let display = advanceLiveDisplay(initial, state(100, [a]));
    for (const at of [200, 1000, 3000, 5099]) {
      display = advanceLiveDisplay(
        display,
        state(at, [{ ...b, lastSupportedAt: at }]),
      );
      expect(display.sign?.id).toBe("a");
    }
    display = advanceLiveDisplay(
      display,
      state(5100, [{ ...b, lastSupportedAt: 5100 }]),
    );
    expect(display.sign?.id).toBe("b");
    expect(display.shownAt).toBe(5100);
  });
  it("expires absent signs and resets on a different camera", () => {
    const display = advanceLiveDisplay(initial, state(100, [a]));
    expect(advanceLiveDisplay(display, state(4999, [])).sign?.lastSeen).toBe(
      true,
    );
    expect(advanceLiveDisplay(display, state(5100, [])).sign).toBeNull();
    expect(
      advanceLiveDisplay(display, { ...state(500, []), source: "stopped" })
        .sign,
    ).toBeNull();
  });
  it("does not reset the hold or animate again for a changed track of the same sign", () => {
    const display = advanceLiveDisplay(initial, state(100, [a]));
    const next = advanceLiveDisplay(
      display,
      state(300, [{ ...a, id: "new-track", lastSupportedAt: 300 }]),
    );
    expect(next.shownAt).toBe(100);
    expect(next.sign?.id).toBe("new-track");
  });
});

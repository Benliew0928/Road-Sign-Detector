import { describe, it, expect, vi } from "vitest";
import manifestJson from "../../public/audio/p16/advisory_audio_manifest.json";
import type { AdvisoryAudioManifest } from "./advisoryAudio";
import { EncounterAudio, type AudioPort } from "./encounterAudio";
import { encounter, sign } from "../test/encounterFixtures";
const manifest = manifestJson as AdvisoryAudioManifest;
function setup(reject?: Error, pack = manifest) {
  const players: AudioPort[] = [];
  const factory = vi.fn(() => {
    const p: AudioPort = {
      play: () => (reject ? Promise.reject(reject) : Promise.resolve()),
      pause: vi.fn(),
      onended: null,
      onerror: null,
    };
    players.push(p);
    return p;
  });
  const report = vi.fn();
  return {
    scheduler: new EncounterAudio(pack, factory, report),
    players,
    factory,
    report,
  };
}
const settle = async () => {
  await Promise.resolve();
  await Promise.resolve();
};
describe("encounter speech scheduler", () => {
  it("speaks once despite hundreds of frames, language changes and ID churn", async () => {
    const { scheduler, factory, players } = setup();
    let e = encounter();
    scheduler.update([e], 0, "en", true);
    await settle();
    players[0].onended?.(new Event("ended"));
    for (let i = 1; i < 300; i++) {
      e = { ...e, lastSupportedAt: i * 100 };
      scheduler.update([e], i * 100, "ms", true);
    }
    scheduler.update([{ ...e, id: "new-track" }], 30000, "zh", true);
    expect(factory).toHaveBeenCalledTimes(1);
  });
  it("rearms after genuine absence and announces changed signatures", async () => {
    const { scheduler, factory, players } = setup();
    scheduler.update([encounter()], 0, "en", true);
    await settle();
    players[0].onended?.(new Event("ended"));
    scheduler.update([], 6000, "en", true);
    scheduler.update([encounter("two", 6100)], 6100, "en", true);
    await settle();
    expect(factory).toHaveBeenCalledTimes(2);
  });
  it("unknown, missing safety approval and blocked actions are silent", () => {
    const { scheduler, factory } = setup();
    const events = [
      sign({ semantic_sign_id: "unknown_sign" }),
      sign({ advisory: undefined }),
      sign({ evidence: ["safety:directional_action_blocked"] }),
      sign({ action: { ...sign().action, code: "UNKNOWN_CAUTION" } }),
    ];
    scheduler.update(
      events.map((e, i) => encounter(String(i), 0, e)),
      0,
      "en",
      true,
    );
    expect(factory).not.toHaveBeenCalled();
  });
  it("only critical events interrupt, and silence invalidates pending callbacks", async () => {
    const { scheduler, factory, players } = setup();
    const low = encounter(
      "low",
      0,
      sign({ semantic_sign_id: "parking", severity: "information" }),
    );
    scheduler.update([low], 0, "en", true);
    await settle();
    scheduler.update(
      [
        low,
        encounter(
          "warning",
          100,
          sign({ semantic_sign_id: "roadworks", severity: "warning" }),
        ),
      ],
      100,
      "en",
      true,
    );
    expect(factory).toHaveBeenCalledTimes(1);
    scheduler.update([low, encounter("stop", 200)], 200, "en", true);
    expect(players[0].pause).toHaveBeenCalled();
    expect(factory).toHaveBeenCalledTimes(2);
    scheduler.update([], 300, "en", false);
    expect(players[1].pause).toHaveBeenCalled();
    expect(players[1].onended).toBeNull();
  });
  it("coalesces simultaneous identical signs and drops stale items", async () => {
    const { scheduler, factory, players } = setup();
    scheduler.update([encounter("a"), encounter("b")], 0, "en", true);
    await settle();
    players[0].onended?.(new Event("ended"));
    scheduler.update([encounter("a"), encounter("b")], 1500, "en", true);
    expect(factory).toHaveBeenCalledTimes(1);
    scheduler.update(
      [encounter("old", 0, sign({ semantic_sign_id: "parking" }))],
      6000,
      "en",
      true,
    );
    expect(factory).toHaveBeenCalledTimes(1);
  });
  it("autoplay denial waits for interaction, with no fallback storm", async () => {
    const { scheduler, factory, report } = setup(
      new DOMException("blocked", "NotAllowedError"),
    );
    scheduler.update([encounter()], 0, "en", true);
    await settle();
    for (let i = 1; i < 20; i++)
      scheduler.update([encounter()], i * 100, "en", true);
    expect(factory).toHaveBeenCalledTimes(1);
    expect(report).toHaveBeenLastCalledWith(expect.any(String), true);
    scheduler.enable();
    await settle();
    expect(factory).toHaveBeenCalledTimes(2);
  });
  it("seeking cancels speech and skips old encounters; new scheduler replays", async () => {
    const { scheduler, factory } = setup();
    scheduler.skipThrough(5000);
    scheduler.update([encounter("old", 1000)], 5000, "en", true);
    expect(factory).not.toHaveBeenCalled();
    scheduler.update([encounter("new", 6000)], 6000, "en", true);
    await settle();
    expect(factory).toHaveBeenCalledTimes(1);
  });
  it("honors the one second gap between queued recordings", async () => {
    const { scheduler, factory, players } = setup();
    const low = encounter(
      "low",
      0,
      sign({ semantic_sign_id: "parking", severity: "information" }),
    );
    scheduler.update([encounter(), low], 0, "en", true);
    await settle();
    players[0].onended?.(new Event("ended"));
    scheduler.update([encounter(), low], 900, "en", true);
    expect(factory).toHaveBeenCalledTimes(1);
    scheduler.update([encounter(), low], 1000, "en", true);
    expect(factory).toHaveBeenCalledTimes(2);
  });
});

it("tries the local fallback only once on asset errors", async () => {
  const pack = structuredClone(manifest);
  const pid = pack.semantic_phrase_ids.stop;
  pack.phrases[pid].assets.en.fallback_src = "/audio/fallback.wav";
  const { scheduler, factory } = setup(new Error("decode failure"), pack);
  scheduler.update([encounter()], 0, "en", true);
  await settle();
  await settle();
  for (let i = 1; i < 100; i++)
    scheduler.update([encounter()], i * 100, "en", true);
  expect(factory).toHaveBeenCalledTimes(2);
});
it("a confirmed numeric revision speaks once and waits for a real-time gap", async () => {
  const { scheduler, factory, players } = setup();
  const e = encounter(
    "speed",
    0,
    sign({
      semantic_sign_id: "maximum_speed",
      action: {
        ...sign().action,
        code: "SET_TARGET_SPEED",
        target_speed_kmh: 50,
      },
    }),
  );
  scheduler.update([e], 0, "en", true, 0);
  await settle();
  players[0].onended?.(new Event("ended"));
  const revised = encounter(
    "speed",
    2000,
    sign({
      semantic_sign_id: "maximum_speed",
      action: { ...e.event.action, target_speed_kmh: 80 },
    }),
  );
  scheduler.update([revised], 2000, "en", true, 500);
  expect(factory).toHaveBeenCalledTimes(1);
  scheduler.update([revised], 3000, "en", true, 1000);
  await settle();
  expect(factory).toHaveBeenCalledTimes(2);
});
it("finishes confirmed speech through a brief unrecognized observation", async () => {
  const { scheduler, players } = setup();
  const e = encounter();
  scheduler.update([e], 0, "en", true);
  await settle();
  scheduler.update([{ ...e, speechEligible: false }], 300, "en", true);
  expect(players[0].pause).not.toHaveBeenCalled();
});

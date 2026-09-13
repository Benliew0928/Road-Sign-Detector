import { describe, expect, it } from "vitest";
import { automaticVideoSign, focusTransform, videoSigns } from "./lens";
import { frame, sign } from "./test/encounterFixtures";
describe("Lens focus", () => {
  it.each([
    { width: 1280, height: 600 },
    { width: 390, height: 550 },
  ])("centers the sign in the clear aperture at $width px", (view) => {
    const box = { x1: 100, y1: 80, x2: 180, y2: 160 };
    const g = focusTransform(view, { width: 640, height: 480 }, box);
    const numbers = g.transform.match(/-?[\d.]+/g)!.map(Number);
    expect(g.left + numbers[0] + ((140 * g.width) / 640) * g.zoom).toBeCloseTo(
      view.width * (view.width < 720 ? 0.5 : 0.4),
    );
    expect(g.top + numbers[1] + ((120 * g.height) / 480) * g.zoom).toBeCloseTo(
      view.height * (view.width < 720 ? 0.32 : 0.46),
    );
    expect(g.zoom).toBeGreaterThan(1);
    expect(g.zoom).toBeLessThanOrEqual(6);
  });
  it("excludes unknown, low confidence and held predictions from automatic details", () => {
    const result = frame(0, [
      sign({ confidence: 0.79 }),
      sign({ semantic_sign_id: "unknown_sign", confidence: 0.99 }),
      sign({ confidence: 0.99, evidence: ["tracker_hold:1"] }),
      sign({ track_id: 4, confidence: 0.9 }),
    ]);
    expect(videoSigns(result)).toHaveLength(1);
    expect(automaticVideoSign(result)?.track_id).toBe(4);
    expect(automaticVideoSign(frame(1, []))).toBeNull();
  });
});

import { describe, expect, it } from "vitest";

import type { FrameResult } from "../types";
import { parseCameraMessage } from "./useCameraStream";

const frameResult: FrameResult = {
  frame_id: 7,
  width: 640,
  height: 360,
  mode: "deep",
  latency_ms: 42,
  events: [],
  warnings: [],
};

describe("parseCameraMessage", () => {
  it("parses frame results and server error payloads", () => {
    expect(parseCameraMessage(JSON.stringify(frameResult))).toEqual(frameResult);
    expect(parseCameraMessage(JSON.stringify({ error: "Frame exceeds 20 MB" }))).toEqual({
      error: "Frame exceeds 20 MB",
    });
  });

  it("rejects malformed camera messages", () => {
    expect(() => parseCameraMessage("{")).toThrow();
    expect(() => parseCameraMessage(new ArrayBuffer(1))).toThrow(
      "Camera response was not text JSON.",
    );
  });
});

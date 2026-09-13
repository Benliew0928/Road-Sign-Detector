import { describe, expect, it } from "vitest";

import type { FrameResult } from "../types";
import { encodeCameraFrame, parseCameraMessage } from "./useCameraStream";

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

  it("parses explicit stale-frame acknowledgements", () => {
    expect(
      parseCameraMessage(
        JSON.stringify({ type: "dropped", frame_seq: 12, reason: "superseded_by_newer_frame" }),
      ),
    ).toEqual({ type: "dropped", frame_seq: 12, reason: "superseded_by_newer_frame" });
  });

  it("adds a sequence envelope without changing the JPEG payload", async () => {
    const payload = new Uint8Array([0xff, 0xd8, 0xff, 0xd9]);
    const framed = encodeCameraFrame(new Blob([payload], { type: "image/jpeg" }), 42);
    const bytes = new Uint8Array(await framed.arrayBuffer());

    expect(Array.from(bytes.slice(0, 4))).toEqual([0x52, 0x53, 0x41, 0x31]);
    expect(new DataView(bytes.buffer).getUint32(4, false)).toBe(42);
    expect(Array.from(bytes.slice(8))).toEqual(Array.from(payload));
  });
});

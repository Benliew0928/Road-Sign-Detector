import {
  act,
  fireEvent,
  render,
  screen,
  cleanup,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { VideoResults } from "./VideoResults";
import { frame, sign } from "../test/encounterFixtures";
import manifest from "../../public/audio/p16/advisory_audio_manifest.json";
let calls: string[] = [];
beforeEach(() => {
  vi.useFakeTimers();
  calls = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(() =>
      Promise.resolve({ ok: true, json: () => Promise.resolve(manifest) }),
    ),
  );
  class FakeAudio {
    src: string;
    currentTime = 0;
    preload = "auto";
    onended: (() => void) | null = null;
    onerror: (() => void) | null = null;
    constructor(src: string) {
      this.src = src;
    }
    play() {
      calls.push(this.src);
      return Promise.resolve();
    }
    pause() {}
  }
  vi.stubGlobal("Audio", FakeAudio);
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(function (
    this: HTMLMediaElement,
  ) {
    this.dispatchEvent(new Event("play"));
    return Promise.resolve();
  });
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
const tick = async () => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(150);
  });
};
it("video speech waits for play, seeking stays silent, and explicit replay rearms", async () => {
  const summary = {
    frames_read: 100,
    sampled_frames: 100,
    events: 100,
    fps: 10,
    event_samples: [],
    representative_result: null,
    frame_results: Array.from({ length: 100 }, (_, i) => ({
      source_frame: i,
      result: frame(i, [sign()]),
    })),
  };
  const { container } = render(
    <VideoResults videoUrl="blob:one" summary={summary} busy={false} />,
  );
  await tick();
  expect(calls).toHaveLength(0);
  expect(
    screen.getByText("Confirmed encounters").nextElementSibling,
  ).toHaveTextContent("1");
  const video = container.querySelector("video")!;
  video.currentTime = 1;
  fireEvent.play(video);
  await tick();
  await tick();
  expect(calls).toHaveLength(1);
  video.currentTime = 2;
  await tick();
  expect(calls).toHaveLength(1);
  fireEvent.pause(video);
  video.currentTime = 0.1;
  fireEvent.seeking(video);
  fireEvent.seeked(video);
  await tick();
  video.currentTime = 1;
  fireEvent.play(video);
  await tick();
  expect(calls).toHaveLength(1);
  fireEvent.click(screen.getByRole("button", { name: "Replay guidance" }));
  await tick();
  video.currentTime = 1;
  await tick();
  await tick();
  expect(calls).toHaveLength(2);
});

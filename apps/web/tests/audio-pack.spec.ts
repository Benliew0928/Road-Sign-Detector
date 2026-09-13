import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";

test.use({
  launchOptions: { args: ["--autoplay-policy=no-user-gesture-required"] },
});
test("repaired assets decode and multilingual recordings reach ended", async ({
  page,
}) => {
  test.setTimeout(60000);
  const report = JSON.parse(
    readFileSync(
      "../../outputs/audit/advisory_audio_repair_20260912.json",
      "utf8",
    ),
  ) as { p16_ai: { repaired: { phrase: string; language: string }[] } };
  const paths = report.p16_ai.repaired.map(
    (e) => `/audio/p16_ai/${e.language}/${e.phrase}.wav`,
  );
  await page.goto("/");
  const durations = await page.evaluate(async (paths) => {
    const context = new AudioContext();
    try {
      return await Promise.all(
        paths.map(async (path) => {
          const response = await fetch(path);
          const buffer = await context.decodeAudioData(
            await response.arrayBuffer(),
          );
          return buffer.duration;
        }),
      );
    } finally {
      await context.close();
    }
  }, paths);
  expect(durations).toHaveLength(43);
  expect(durations.every((d) => d > 0 && d < 30)).toBe(true);
  for (const language of ["en", "ms", "zh"]) {
    const duration = await page.evaluate(async (language) => {
      const audio = new Audio(
        `/audio/p16_ai/${language}/width_restriction.wav`,
      );
      audio.playbackRate = 16;
      return await new Promise<number>((resolve, reject) => {
        const timer = window.setTimeout(() => {
          audio.pause();
          reject(new Error("Audio never ended"));
        }, 15000);
        audio.onended = () => {
          clearTimeout(timer);
          resolve(audio.duration);
        };
        audio.onerror = () => {
          clearTimeout(timer);
          reject(new Error("Audio decode failed"));
        };
        void audio.play().catch(reject);
      });
    }, language);
    expect(duration).toBeGreaterThan(0);
    expect(duration).toBeLessThan(30);
  }
});

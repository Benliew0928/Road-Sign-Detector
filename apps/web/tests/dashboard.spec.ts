import { expect, test } from "@playwright/test";

let phoneStreamSnapshots: unknown[] = [];

const baselineHealth = {
  status: "ok",
  version: "0.1.0",
  diagnostics: {
    python: "3.11",
    opencv: "4.13",
    cuda_available: false,
    official_image_count: 84,
    healthy: true,
  },
  models: {
    runtime_badge: "LEGACY",
    config_name: "legacy_test",
    config_path: "configs/inference/legacy.yaml",
    config_sha256:
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    preprocessing_version: "roadsign_raw_bgr_v1",
    bundle_identity: {},
    mode: "baseline",
    detector: "color_shape_baseline",
    detector_available: true,
    detector_loaded: true,
    detector_device: null,
    classifier: "unavailable",
    classifier_available: false,
    classifier_loaded: false,
    classifier_providers: [],
    tracker: "iou+sparseOptFlow-gmc",
    ocr_available: false,
    ocr_loaded: false,
    ocr_load_error: null,
    warnings: ["Semantic classifier weights are unavailable."],
  },
};

const sampleVideoEvent = {
  schema_version: "1.0",
  frame_id: 12,
  track_id: 4,
  coursework_id: null,
  semantic_sign_id: "side_road_right",
  meaning: {
    en: "Side road right",
    ms: "Simpang sebelah kanan",
    zh: "Right side road",
  },
  ocr: {
    text: "",
    confidence: 0,
    script: "none",
    language: "none",
    numeric_value: null,
    unit: null,
    semantic_sign_id: null,
  },
  confidence: 0.88,
  bbox: { x1: 100, y1: 80, x2: 180, y2: 160 },
  mask: null,
  action: {
    code: "monitor_road",
    target_speed_kmh: null,
    restriction_value: null,
    restriction_unit: null,
    direction: "right",
    advisory_only: true,
  },
  severity: "caution",
  latency_ms: 14.2,
  device: "cpu",
  stable: true,
  should_announce: true,
  evidence: ["video sample"],
  advisory: {
    headline: { en: "Side road", ms: "Side road", zh: "Side road" },
    instruction: {
      en: "Watch for merging traffic.",
      ms: "Watch for merging traffic.",
      zh: "Watch for merging traffic.",
    },
    safe_to_announce: true,
  },
};

test.beforeEach(async ({ page }) => {
  phoneStreamSnapshots = [];
  await page.route(
    "**/audio/p16/advisory_audio_manifest.json",
    async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          schema_version: "test",
          catalogue_version: "test",
          languages: ["en", "ms", "zh"],
          description: "test",
          fallback_phrase_id: "unknown_sign",
          semantic_phrase_ids: { side_road_right: "side_road_right" },
          audio_key_phrase_ids: {},
          variant_phrase_ids: {
            speed_limit_kmh: {},
            minimum_speed_kmh: {},
            temporary_speed_limit_kmh: {},
            height_limit_m: {},
            width_limit_m: {},
            weight_limit_t: {},
          },
          phrases: {
            side_road_right: {
              phrase_id: "side_road_right",
              semantic_sign_id: "side_road_right",
              audio_key: "side_road_right",
              base_action: "WATCH_ROAD_HAZARD",
              severity: "caution",
              priority: 2,
              interrupts_lower_priority: false,
              cooldown_seconds: 8,
              parameter: null,
              text: {
                en: "Side road advice",
                ms: "Nasihat simpang",
                zh: "支路提醒",
              },
              assets: {
                en: {
                  src: "/audio/test.wav",
                  sha256: null,
                  bytes: null,
                  duration_seconds: null,
                  voice: null,
                  generated: true,
                },
                ms: {
                  src: "/audio/test.wav",
                  sha256: null,
                  bytes: null,
                  duration_seconds: null,
                  voice: null,
                  generated: true,
                },
                zh: {
                  src: "/audio/test.wav",
                  sha256: null,
                  bytes: null,
                  duration_seconds: null,
                  voice: null,
                  generated: true,
                },
              },
            },
            unknown_sign: {
              phrase_id: "unknown_sign",
              semantic_sign_id: "unknown_sign",
              audio_key: "unknown_sign",
              base_action: "UNKNOWN_CAUTION",
              severity: "caution",
              priority: 2,
              interrupts_lower_priority: false,
              cooldown_seconds: 8,
              parameter: null,
              text: { en: "Unknown advice", ms: "Tidak pasti", zh: "未知提醒" },
              assets: {
                en: {
                  src: "/audio/test.wav",
                  sha256: null,
                  bytes: null,
                  duration_seconds: null,
                  voice: null,
                  generated: true,
                },
                ms: {
                  src: "/audio/test.wav",
                  sha256: null,
                  bytes: null,
                  duration_seconds: null,
                  voice: null,
                  generated: true,
                },
                zh: {
                  src: "/audio/test.wav",
                  sha256: null,
                  bytes: null,
                  duration_seconds: null,
                  voice: null,
                  generated: true,
                },
              },
            },
          },
        }),
      });
    },
  );
  await page.route("**/api/v1/health", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(baselineHealth),
    });
  });
  await page.route("**/api/v1/phone/connection", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        session_id: "phone-session",
        phone_url: "https://192.168.1.20:8443/phone?session=phone-session",
        websocket_url: "wss://192.168.1.20:8443/api/v1/ws/camera/phone-session",
        candidate_urls: [
          "https://192.168.1.20:8443/phone?session=phone-session",
        ],
        https: true,
        camera_requires_https: true,
        mode: "local",
        public_base_url: null,
        access_token: null,
        operator_live_url: "/live",
      }),
    });
  });
  await page.route("**/api/v1/phone/streams", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ streams: phoneStreamSnapshots }),
    });
  });
});

const imageFile = (name = "road-sign.png") => ({
  name,
  mimeType: "image/png",
  buffer: Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z4p8AAAAASUVORK5CYII=",
    "base64",
  ),
});
const result = (events = [sampleVideoEvent]) => ({
  frame_id: 0,
  width: 640,
  height: 480,
  mode: "deep",
  latency_ms: 14.2,
  events,
  warnings: [],
});
const screenshotRoot = "../../outputs/lens-20260913";

test("dark local workspace, focus mode and page motion fit desktop and phone", async ({
  page,
}, info) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "RoadSign Assist" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Every sign tells a story." }),
  ).toBeVisible();
  await expect(page.locator(".runtime-badge")).toHaveText("LEGACY");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: `${screenshotRoot}/${info.project.name}-analyze.png`,
    fullPage: true,
    animations: "disabled",
  });
  if (info.project.name === "desktop") {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.screenshot({
      path: `${screenshotRoot}/desktop-1920-home.png`,
      animations: "disabled",
    });
    expect(
      await page
        .locator(".immersive-home h3")
        .evaluate((el) => parseFloat(getComputedStyle(el).fontSize)),
    ).toBeGreaterThanOrEqual(90);
  }
  await page.getByRole("button", { name: "Recent", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "A fresh perspective." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Images", exact: true }).click();
  await page.getByRole("button", { name: "Presenter mode" }).click();
  await expect(
    page.getByRole("button", { name: "Exit presenter mode" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(await page.getByText(/sign in|log in|create account/i).count()).toBe(
    0,
  );
});

test("runtime details stay available and offline service can recover", async ({
  page,
}, info) => {
  let online = false;
  await page.route("**/api/v1/health", (route) =>
    route.fulfill({
      status: online ? 200 : 503,
      contentType: "application/json",
      body: JSON.stringify(
        online
          ? baselineHealth
          : { detail: "Backend intentionally unavailable" },
      ),
    }),
  );
  await page.goto("/");
  await expect(
    page.getByText("Backend intentionally unavailable"),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Add media", exact: true }),
  ).toBeDisabled();
  online = true;
  await page.getByRole("button", { name: "Reconnect", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Add media", exact: true }),
  ).toBeEnabled();
  await page.locator(".system-details > summary").click();
  await expect(page.locator(".system-popover")).toBeVisible();
  expect(
    await page
      .locator(".system-popover")
      .evaluate((el) => el.scrollWidth <= el.clientWidth),
  ).toBe(true);
  await page.screenshot({
    path: `${screenshotRoot}/${info.project.name}-system-compact.png`,
    animations: "disabled",
  });
  await expect(
    page.getByText("Classical baseline", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Refresh status" }),
  ).toBeVisible();
});

test("image orientation, selectable findings and original overlay toggle", async ({
  page,
}, info) => {
  let calls = 0;
  await page.route("**/api/v1/infer/image", (route) => {
    calls++;
    return route.fulfill({
      json: {
        result: result([
          sampleVideoEvent,
          {
            ...sampleVideoEvent,
            track_id: 5,
            bbox: { x1: 300, y1: 80, x2: 380, y2: 160 },
            semantic_sign_id: "unknown_sign",
            confidence: 0,
          },
        ]),
        annotated_jpeg_base64: "",
      },
    });
  });
  await page.goto("/");
  await page.locator('input[accept^="image/"]').setInputFiles(imageFile());
  await expect(
    page.getByRole("heading", { name: "Make sure the road scene is upright" }),
  ).toBeVisible();
  expect(calls).toBe(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollHeight <= innerHeight,
    ),
  ).toBe(true);
  await page.screenshot({
    path: `${screenshotRoot}/${info.project.name}-preparation-fixed.png`,
    animations: "disabled",
  });

  await page.getByRole("button", { name: "Rotate right" }).click();
  await expect(
    page.getByAltText("Selected road scene awaiting orientation confirmation"),
  ).toHaveCSS("transform", "matrix(0, 1, -1, 0, 0, 0)");
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await page.getByRole("button", { name: "Analyze upright image" }).click();
  await expect(page.locator(".lens-sign-box")).toHaveCount(2);
  const canvas = await page.locator(".lens-canvas").boundingBox();
  expect(canvas?.x).toBe(0);
  expect(canvas?.y).toBe(0);
  expect(canvas?.width).toBe(page.viewportSize()!.width);
  expect(canvas?.height).toBe(page.viewportSize()!.height);

  await page.getByRole("button", { name: "Full image", exact: true }).click();
  await page.locator(".lens-sign-box").first().click();
  await expect(page.locator(".lens-focus-atmosphere")).toHaveClass(/active/);
  await page.screenshot({
    path: `${screenshotRoot}/${info.project.name}-focused.png`,
    animations: "disabled",
  });
  await expect(page.locator(".lens-details h2")).toHaveText("Side road right");
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page.locator(".lens-sign-box").nth(1).click();
  await expect(
    page.getByText("Meaning uncertain", { exact: true }),
  ).toBeVisible();
  await expect(
    page.locator(".lens-details").getByText("0%", { exact: true }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page.getByRole("button", { name: "Original", exact: true }).click();
  await expect(page.locator(".lens-sign-box")).toHaveCount(0);
  await page.getByRole("button", { name: "Result", exact: true }).click();
  await expect(page.locator(".lens-sign-box")).toHaveCount(2);
  expect(calls).toBe(1);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: `${screenshotRoot}/${info.project.name}-result.png`,
    fullPage: true,
    animations: "disabled",
  });
});

test("mixed batch preserves per-image type, progress and recent provenance", async ({
  page,
}, info) => {
  let roadCalls = 0;
  let closeCalls = 0;
  await page.route("**/api/v1/infer/image", (route) => {
    roadCalls++;
    return route.fulfill({
      json: { result: result(), annotated_jpeg_base64: "" },
    });
  });
  await page.route("**/api/v1/infer/close-up", (route) => {
    closeCalls++;
    return route.fulfill({
      json: { result: result(), annotated_jpeg_base64: "" },
    });
  });
  await page.goto("/");
  await page
    .locator('input[accept^="image/"]')
    .setInputFiles([imageFile("scene.png"), imageFile("close.png")]);
  await page.getByRole("button", { name: "Prepare close.png" }).click();
  await page.getByRole("button", { name: /Close-up sign/ }).click();
  await page.getByRole("button", { name: "Analyze 2 images" }).click();
  await expect(
    page.getByRole("table", { name: "Batch inference results" }),
  ).toBeVisible();
  expect(roadCalls).toBe(1);
  expect(closeCalls).toBe(1);
  await page
    .getByRole("button", { name: "Open details for close.png" })
    .click();
  await expect(page.locator(".context-toolbar .type-badge")).toHaveText(
    "Close-up sign",
  );
  await page.getByRole("button", { name: "Back to batch results" }).click();
  await expect(
    page.getByRole("table", { name: "Batch inference results" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Recent", exact: true }).click();
  await page.locator(".recent-card").filter({ hasText: "close.png" }).click();
  await expect(page.locator(".context-toolbar .type-badge")).toHaveText(
    "Close-up sign",
  );
  await page.locator(".lens-image-evidence summary").click();
  expect(
    await page
      .locator(".analysis-popover")
      .evaluate((el) => el.scrollWidth <= el.clientWidth),
  ).toBe(true);
  await page.screenshot({
    path: `${screenshotRoot}/${info.project.name}-analysis-compact.png`,
    animations: "disabled",
  });
  await expect(
    page.getByText("Whole-image scan", {
      exact: true,
    }),
  ).toBeVisible();
});

test("failed batch items can retry without rerunning successful images", async ({
  page,
}) => {
  let calls = 0;
  await page.route("**/api/v1/infer/image", (route) => {
    calls++;
    return route.fulfill(
      calls === 2
        ? { status: 422, json: { detail: "Cannot decode this image" } }
        : { json: { result: result(), annotated_jpeg_base64: "" } },
    );
  });
  await page.goto("/");
  await page
    .locator('input[accept^="image/"]')
    .setInputFiles([imageFile("good.png"), imageFile("retry.png")]);
  await page.getByRole("button", { name: "Analyze 2 images" }).click();
  await expect(page.getByText("Cannot decode this image")).toBeVisible();
  await page.getByRole("button", { name: "Retry failed images" }).click();
  await expect(
    page.getByRole("button", { name: "Open details for retry.png" }),
  ).toBeEnabled();
  expect(calls).toBe(3);
});

test("video returns synchronized findings and errors never read as complete", async ({
  page,
}, info) => {
  await page.route("**/api/v1/infer/video?*", (route) =>
    route.fulfill({
      json: {
        frames_read: 90,
        sampled_frames: 90,
        events: 4,
        fps: 30,
        frame_results: Array.from({ length: 90 }, (_, i) => ({
          source_frame: i,
          result: { ...result(), frame_id: i },
        })),
        event_samples: [sampleVideoEvent],
        representative_result: result(),
      },
    }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Video", exact: true }).click();
  await page.locator('input[accept^="video/"]').setInputFiles({
    name: "road.mp4",
    mimeType: "video/mp4",
    buffer: Buffer.from("fixture"),
  });
  await expect(
    page.getByText("Analysis complete", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".lens-details")).toHaveCount(0);
  await page.getByRole("button", { name: "Full image", exact: true }).click();
  await page.getByRole("button", { name: "Inspect Side road right" }).click();
  await expect(page.locator(".lens-focus-atmosphere")).toHaveClass(/active/);
  await page.locator("video").dispatchEvent("play");
  await expect(page.locator(".lens-focus-atmosphere")).not.toHaveClass(
    /active/,
  );
  await expect(
    page.getByRole("button", { name: "Inspect Side road right" }),
  ).toHaveCount(0);
  await page.locator("video").evaluate((video) =>
    Object.defineProperty(video, "currentTime", {
      configurable: true,
      writable: true,
      value: 1,
    }),
  );
  await expect(page.locator(".lens-details h2")).toHaveText("Side road right");
  await page
    .getByRole("combobox", { name: "Warning language" })
    .selectOption("ms");
  await expect(page.locator(".lens-details h2")).toHaveText(
    "Simpang sebelah kanan",
  );
  await expect(
    page
      .locator(".video-summary dl > div")
      .filter({ hasText: "Confirmed encounters" })
      .locator("dd"),
  ).toHaveText("1");
  await page.screenshot({
    path: `../../outputs/encounter-ux/${info.project.name}-video.png`,
    fullPage: true,
  });
  await page.route("**/api/v1/infer/video?*", (route) =>
    route.fulfill({ status: 400, json: { detail: "Unable to decode video" } }),
  );
  await page.locator('input[accept^="video/"]').setInputFiles({
    name: "bad.mp4",
    mimeType: "video/mp4",
    buffer: Buffer.from("invalid"),
  });
  await expect(
    page.getByText("Analysis failed", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Analysis complete", { exact: true }),
  ).toHaveCount(0);
});

test("phone pairing and integrated live wall keep camera focus accessible", async ({
  page,
}, info) => {
  phoneStreamSnapshots = [
    {
      stream_id: "one",
      session_id: "phone-session",
      label: "Device 1",
      connected_at: 1,
      updated_at: Date.now() / 1000,
      frame_seq: 3,
      width: 640,
      height: 480,
      jpeg_base64: imageFile().buffer.toString("base64"),
      live_fps: 30,
      inference_fps: 8.5,
      inference_pending: false,
      inference_frame_seq: 3,
      result: result(),
    },
  ];
  await page.routeWebSocket("**/api/v1/ws/phone/monitor*", (socket) =>
    socket.send(
      JSON.stringify({ type: "snapshot", streams: phoneStreamSnapshots }),
    ),
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Live", exact: true }).click();
  await page.getByRole("button", { name: "Add phone", exact: true }).click();
  await expect(
    page.getByAltText("Phone camera connection QR code"),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Open host live camera wall" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: `${screenshotRoot}/${info.project.name}-pairing.png`,
    fullPage: true,
    animations: "disabled",
  });
  await page.getByRole("button", { name: "Zoom Device 1" }).click();
  await expect(
    page.getByRole("dialog", { name: "Camera zoom view" }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Close zoom" })).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(page.getByRole("button", { name: "Next camera" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("button", { name: "Zoom Device 1" }),
  ).toBeFocused();
});

test("stale and mismatched phone frames do not display current detection boxes", async ({
  page,
}) => {
  const streams = [
    {
      stream_id: "one",
      label: "Stale phone",
      session_id: "phone-session",
      connected_at: 1,
      updated_at: Date.now() / 1000 - 60,
      frame_seq: 8,
      width: 640,
      height: 480,
      jpeg_base64: imageFile().buffer.toString("base64"),
      live_fps: 0,
      inference_fps: 0,
      inference_pending: true,
      inference_frame_seq: 3,
      result: result(),
    },
  ];
  await page.routeWebSocket("**/api/v1/ws/phone/monitor*", (socket) =>
    socket.send(JSON.stringify({ type: "snapshot", streams })),
  );
  await page.goto("/live");
  await expect(page.locator(".feed-state")).toContainText("Last frame");
  await expect(page.locator(".live-detection-box")).toHaveCount(0);
  await expect(page.locator(".live-camera-meta")).toContainText(
    "No confirmed sign",
  );
});

test("phone sender puts camera controls before collapsed diagnostics", async ({
  page,
}, info) => {
  await page.goto("/phone?session=phone-session");
  await expect(
    page.getByRole("button", { name: "Start stream", exact: true }),
  ).toBeVisible();
  await expect(page.getByLabel("Resolution")).toBeHidden();
  await page
    .getByRole("button", { name: "Camera settings", exact: true })
    .click();
  await expect(page.getByLabel("Resolution")).toHaveValue("4096");
  await page.getByRole("button", { name: "Close camera settings" }).click();
  await expect(
    page.getByRole("navigation", { name: "Main navigation" }),
  ).toHaveCount(0);
  await expect(
    page.locator(".phone-events,.stream-details,.phone-detection-box"),
  ).toHaveCount(0);
  expect(await page.locator(".phone-live-stage").boundingBox()).toEqual({
    x: 0,
    y: 0,
    width: page.viewportSize()!.width,
    height: page.viewportSize()!.height,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: `${screenshotRoot}/${info.project.name}-sender.png`,
    fullPage: true,
    animations: "disabled",
  });
});

test("iPad portrait, landscape and split-screen have no overflow and honor reduced motion", async ({
  page,
}, info) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const [width, height] of [
    [768, 1024],
    [1180, 820],
    [500, 820],
  ]) {
    await page.setViewportSize({ width, height });
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Every sign tells a story." }),
    ).toBeVisible();
    expect(
      await page
        .locator(".activity-transition")
        .evaluate((element) => getComputedStyle(element).animationName),
    ).toBe("none");
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    if (info.project.name === "desktop")
      await page.screenshot({
        path: `${screenshotRoot}/tablet-${width}.png`,
        fullPage: true,
        animations: "disabled",
      });
  }
});

test("live findings stay confirmed and only the selected feed speaks", async ({
  page,
}, info) => {
  await page.addInitScript(() => {
    const host = window as unknown as Window & { guidanceCalls: string[] };
    host.guidanceCalls = [];
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
        host.guidanceCalls.push(this.src);
        setTimeout(() => this.onended?.(), 10);
        return Promise.resolve();
      }
      pause() {}
    }
    window.Audio = FakeAudio as unknown as typeof Audio;
  });
  let ticks = 0;
  let timer: ReturnType<typeof setInterval> | undefined;
  await page.routeWebSocket("**/api/v1/ws/phone/monitor*", (socket) => {
    const send = () => {
      const streams = ["one", "two"].map((id, index) => ({
        stream_id: id,
        label: `Phone ${id}`,
        session_id: id,
        device_id: id,
        connected_at: index + 1,
        updated_at: Date.now() / 1000,
        frame_seq: ticks,
        width: 640,
        height: 480,
        jpeg_base64: imageFile().buffer.toString("base64"),
        live_fps: 10,
        inference_fps: 10,
        inference_pending: false,
        inference_frame_seq: ticks,
        result: {
          ...result([
            {
              ...sampleVideoEvent,
              advisory: {
                headline: sampleVideoEvent.meaning,
                instruction: sampleVideoEvent.meaning,
                safe_to_announce: true,
              },
            },
          ]),
          frame_id: ticks,
        },
      }));
      socket.send(JSON.stringify({ type: "snapshot", streams }));
      ticks++;
    };
    send();
    timer = setInterval(send, 100);
  });
  try {
    await page.goto("/live");
    await expect(
      page.locator(".live-wall-shell > .encounter-panel h2"),
    ).toHaveText("Side road right");
    const count = () =>
      page.evaluate(
        () =>
          (window as unknown as Window & { guidanceCalls: string[] })
            .guidanceCalls.length,
      );
    await expect.poll(count).toBe(1);
    await expect.poll(() => ticks).toBeGreaterThan(20);
    expect(await count()).toBe(1);
    await page.getByRole("button", { name: "Zoom Phone two" }).click();
    await expect.poll(count).toBe(2);
    await page.getByRole("button", { name: "Close zoom" }).click();
    await expect(
      page.getByText("Guidance: Phone two", { exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Mute guidance", exact: true })
      .click();
    await expect(
      page.getByRole("button", { name: "Unmute guidance", exact: true }),
    ).toBeVisible();
    await page.screenshot({
      path: `../../outputs/encounter-ux/${info.project.name}-live.png`,
      fullPage: true,
    });
  } finally {
    if (timer) clearInterval(timer);
  }
});

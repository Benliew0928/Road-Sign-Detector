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
};

test.beforeEach(async ({ page }) => {
  phoneStreamSnapshots = [];
  await page.route("**/audio/p16/advisory_audio_manifest.json", async (route) => {
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
            text: { en: "Side road advice", ms: "Nasihat simpang", zh: "支路提醒" },
            assets: {
              en: { src: "/audio/test.wav", sha256: null, bytes: null, duration_seconds: null, voice: null, generated: true },
              ms: { src: "/audio/test.wav", sha256: null, bytes: null, duration_seconds: null, voice: null, generated: true },
              zh: { src: "/audio/test.wav", sha256: null, bytes: null, duration_seconds: null, voice: null, generated: true },
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
              en: { src: "/audio/test.wav", sha256: null, bytes: null, duration_seconds: null, voice: null, generated: true },
              ms: { src: "/audio/test.wav", sha256: null, bytes: null, duration_seconds: null, voice: null, generated: true },
              zh: { src: "/audio/test.wav", sha256: null, bytes: null, duration_seconds: null, voice: null, generated: true },
            },
          },
        },
      }),
    });
  });
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
        candidate_urls: ["https://192.168.1.20:8443/phone?session=phone-session"],
        https: true,
        camera_requires_https: true,
        mode: "local",
        public_base_url: null,
        access_token: null,
        operator_live_url: null,
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

test("dashboard has no horizontal overflow and supports presenter mode", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "RoadSign Assist" })).toBeVisible();
  await expect(page.getByText("System ready")).toBeVisible();
  await expect(page.getByRole("button", { name: "Mute warnings" })).toBeVisible();
  await page.getByRole("button", { name: "Presenter mode" }).click();
  await expect(page.getByRole("button", { name: "Exit presenter mode" })).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  expect(overflow).toBe(false);
});

test("dashboard shows deep runtime diagnostics from health", async ({ page }) => {
  await page.route("**/api/v1/health", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        ...baselineHealth,
        models: {
          ...baselineHealth.models,
          mode: "deep",
          detector: "hybrid:ultralytics:emtd_segmenter_s30.onnx+color_shape_baseline",
          detector_available: true,
          detector_loaded: true,
          detector_device: "cuda:0",
          classifier: "onnx:emtd_classifier_efficientnet15_embedding.onnx",
          classifier_available: true,
          classifier_loaded: true,
          classifier_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"],
          ocr_available: true,
          ocr_loaded: true,
          warnings: ["Experimental unreviewed models are active; results are not production claims."],
        },
      }),
    });
  });
  await page.goto("/");
  await expect(page.getByText("System ready")).toBeVisible();
  await expect(page.getByText("Semantic AI pipeline")).toBeVisible();
  await expect(page.getByText("Development mode")).toBeVisible();
  const bodyText = await page.evaluate(() => document.body.innerText);
  expect(bodyText).toMatch(/cuda:0/i);
  expect(bodyText).toContain("CUDAExecutionProvider, CPUExecutionProvider");
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  expect(overflow).toBe(false);
});

test("dashboard disables actions while backend is offline and recovers on refresh", async ({
  page,
}) => {
  let online = false;
  await page.route("**/api/v1/health", async (route) => {
    if (!online) {
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Backend intentionally unavailable" }),
      });
      return;
    }
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(baselineHealth),
    });
  });
  await page.goto("/");
  await expect(page.getByText("Backend offline")).toBeVisible();
  await expect(page.getByText("Backend intentionally unavailable")).toBeVisible();
  await expect(page.getByRole("button", { name: "Choose image" })).toBeDisabled();

  online = true;
  await page.getByRole("button", { name: "Refresh status" }).click();
  await expect(page.getByText("System ready")).toBeVisible();
  await expect(page.getByRole("button", { name: "Choose image" })).toBeEnabled();
});

test("batch workflow renders inference results", async ({ page }) => {
  await page.route("**/api/v1/infer/batch", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        count: 1,
        results: [
          {
            filename: "sample.png",
            result: {
              frame_id: 0,
              width: 80,
              height: 80,
              mode: "baseline",
              latency_ms: 7.4,
              events: [],
              warnings: [],
            },
          },
        ],
      }),
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Batch" }).click();
  await page.locator('input[type="file"][multiple]').setInputFiles({
    name: "sample.png",
    mimeType: "image/png",
    buffer: Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z4p8AAAAASUVORK5CYII=",
      "base64",
    ),
  });
  await expect(page.getByRole("table", { name: "Batch inference results" })).toBeVisible();
  await expect(page.getByText("No sign detected")).toBeVisible();
});

test("video workflow renders processing summary", async ({ page }) => {
  await page.route("**/api/v1/infer/video", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        frames_read: 90,
        sampled_frames: 30,
        events: 4,
        event_samples: [sampleVideoEvent],
        representative_result: {
          frame_id: 12,
          width: 640,
          height: 480,
          mode: "baseline",
          latency_ms: 14.2,
          events: [sampleVideoEvent],
          warnings: [],
        },
      }),
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Video" }).click();
  await page.locator('input[type="file"][accept^="video/"]').setInputFiles({
    name: "road.mp4",
    mimeType: "video/mp4",
    buffer: Buffer.from("test-video"),
  });
  await expect(page.getByText("Analysis complete")).toBeVisible();
  await expect(page.getByText("90")).toBeVisible();
  await expect(page.getByText("30")).toBeVisible();
  await expect(page.locator(".video-summary").getByText("4", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Side road right" })).toBeVisible();
});

test("phone workflow renders QR connection details", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Phone" }).click();
  await expect(page.getByRole("heading", { name: "Scan to stream from your phone" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Open host live camera wall" })).toBeVisible();
  await expect(page.getByAltText("Phone camera connection QR code")).toBeVisible();
  await expect(
    page.locator(".phone-link-box").getByText("https://192.168.1.20:8443/phone?session=phone-session"),
  ).toBeVisible();
});

test("host live camera wall opens from the QR page and supports zoom", async ({ page }) => {
  phoneStreamSnapshots = [
    {
      stream_id: "phone-stream-1",
      session_id: "phone-session",
      label: "Device 1",
      connected_at: 1,
      updated_at: Date.now() / 1000,
      frame_seq: 3,
      width: 80,
      height: 80,
      jpeg_base64:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z4p8AAAAASUVORK5CYII=",
      live_fps: 60,
      inference_fps: 8.5,
      inference_pending: false,
      inference_frame_seq: 3,
      result: {
        frame_id: 3,
        width: 80,
        height: 80,
        mode: "baseline",
        latency_ms: 10,
        events: [sampleVideoEvent],
        warnings: [],
      },
    },
  ];

  await page.goto("/");
  await page.getByRole("button", { name: "Phone" }).click();
  await page.getByRole("link", { name: "Open host live camera wall" }).click();
  await expect(page).toHaveURL(/\/live$/);
  await expect(page.getByLabel("Connected phone camera live footage")).toBeVisible();
  await expect(page.getByRole("button", { name: "Zoom Device 1" })).toBeVisible();

  await page.getByRole("button", { name: "Zoom Device 1" }).click();
  await expect(page.getByRole("dialog", { name: "Camera zoom view" })).toBeVisible();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("a");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog", { name: "Camera zoom view" })).toHaveCount(0);
});

test("phone sender route renders camera controls", async ({ page }) => {
  await page.goto("/phone?session=phone-session");
  await expect(page.getByRole("heading", { name: "RoadSign Assist" })).toBeVisible();
  await expect(page.getByText("Phone camera link")).toBeVisible();
  await expect(page.getByRole("button", { name: "Start stream", exact: true })).toBeVisible();
  await expect(page.getByLabel("Phone camera controls")).toContainText("Rear camera");
  await expect(page.getByLabel("Phone stream metrics")).toContainText("Quality");
});
